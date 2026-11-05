"""Translate a PubMed query into NeuroStore's search syntax.

NeuroStore matches whole words after English stemming, in titles and abstracts (and, with
``search_full_text``, in the paper's text). It understands quoted phrases, AND, OR, NOT, a
leading ``-`` and parentheses. PubMed queries carry things it cannot express, so each is
mapped as below and every lossy step is reported back, never applied silently:

- **Field tags** for text (``[tiab]``, ``[Title/Abstract]``, ``[tw]`` ...) are removed: the
  term searches titles and abstracts, plus full text when that is on.
- **MeSH** (``[mh]``, ``[Mesh]``) has no counterpart. A heading inside an OR group becomes the
  heading's words as a phrase, one alternative among others. A heading that is AND-ed onto
  the query (``AND Humans[mh]``) is a filter on indexing, and is dropped: as a phrase it would
  demand the word appear in the text.
- **Publication type, language and date tags** (``[pt]``, ``[lang]``, ``[dp]``) are dropped.
  Dates come from the search config's ``date_from``/``date_to`` as a year range instead.
- **Truncation** (``reactiv*``) has no counterpart either, because matching is on whole
  stemmed words. A truncated word becomes an OR of the prefix with common English endings,
  which the server's stemmer folds onto the words the truncation meant to reach.

PubMed evaluates booleans left to right, so the output is fully parenthesized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

TEXT_TAGS = {
    "", "tiab", "title/abstract", "ti", "title", "ab", "abstract", "tw", "textword",
    "text word", "all", "allfields", "all fields", "word",
}
MESH_TAGS = {
    "mh", "mesh", "meshterms", "mesh terms", "majr", "mesh major topic", "sh", "subheading",
    "mh:noexp", "mesh:noexp", "mesh terms:noexp",
}
DROP_TAGS = {
    "pt", "publication type", "publicationtype", "la", "lang", "language", "dp",
    "date - publication", "publication date", "pdat", "edat", "entrez date", "crdt",
    "mhda", "sb", "subset", "filter", "jour", "journal", "ta", "au", "author",
}
# Endings tried after a truncated prefix. Not a dictionary: a combination that is not an
# English word matches nothing, so over-generating costs nothing but query length.
# The stemmer already folds plurals and -ed/-ing onto the prefix's own stem, so only endings
# that change the stem are worth their query length: densit* -> density, meta-analy* ->
# meta-analysis.
TRUNCATION_ENDINGS = ("", "e", "y", "ion", "ity", "sis", "ic")
MIN_TRUNCATION_PREFIX = 3


@dataclass
class Translation:
    query: str
    dropped: List[str] = field(default_factory=list)  # clauses removed, as written
    approximated: List[str] = field(default_factory=list)  # clauses rewritten lossily
    # The query split at its top-level NOTs: `query` is `include` NOT each of `exclude`.
    # NeuroStore applies NOT to every field it searches, so with full text on, a PubMed
    # `NOT review[tiab]` would drop every paper whose text mentions a review. A full-text
    # search runs `include` and subtracts a title-and-abstract search of `exclude`.
    include: str = ""
    exclude: List[str] = field(default_factory=list)


class QueryTranslationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# tokens and AST
# ---------------------------------------------------------------------------

@dataclass
class Leaf:
    words: List[str]  # one word, or the words of a phrase
    phrase: bool
    tag: str  # normalized, "" when untagged
    raw: str


@dataclass
class Range:
    low: Leaf
    high: Leaf

    @property
    def raw(self) -> str:
        return f"{self.low.raw} : {self.high.raw}"


@dataclass
class Op:
    op: str  # "AND" | "OR" | "NOT"
    left: "Node"
    right: "Node"


Node = Union[Leaf, Range, Op]
_TOKEN = re.compile(
    r'\s*(?:(?P<lp>\()|(?P<rp>\))|(?P<colon>:)|(?P<tag>\[[^\]]*\])'
    r'|(?P<dq>"[^"]*")|(?P<sq>\'[^\']*\'(?=\s|\[|\)|$))|(?P<word>[^\s()\[\]":]+))'
)


def _tokens(query: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pos = 0
    query = query.strip()
    while pos < len(query):
        m = _TOKEN.match(query, pos)
        if not m or m.end() == pos:
            if query[pos:].strip() == "":
                break
            raise QueryTranslationError(f"cannot read the query at: {query[pos:pos + 30]!r}")
        pos = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        if kind in ("dq", "sq"):
            out.append(("phrase", value[1:-1]))
        elif kind == "word" and value in ("AND", "OR", "NOT"):
            out.append(("op", value))
        else:
            out.append((kind, value))
    return out


def _normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip("[]").strip().lower())


class _Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.i = 0
        self.notes: List[str] = []

    def peek(self) -> Optional[Tuple[str, str]]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> Tuple[str, str]:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def parse(self) -> Node:
        node = self.expression()
        # PubMed ignores a surplus closing parenthesis, so a query that ran there carries one
        # unnoticed. Accept it too, but say so.
        while self.peek() is not None and self.peek()[0] == "rp":
            self.take()
            self.notes.append("an unmatched ')' in the query was ignored")
            more = self.peek()
            if more is not None:
                if more[0] == "op":
                    self.take()
                    node = Op(more[1], node, self.operand())
                else:
                    node = Op("AND", node, self.operand())
                while self.peek() is not None and self.peek()[0] != "rp":
                    token = self.take() if self.peek()[0] == "op" else ("op", "AND")
                    node = Op(token[1], node, self.operand())
        if self.peek() is not None:
            raise QueryTranslationError(f"unexpected {self.peek()[1]!r}")
        return node

    def expression(self) -> Node:
        # PubMed: left to right, no precedence. Juxtaposed operands mean AND.
        node = self.operand()
        while True:
            token = self.peek()
            if token is None or token[0] == "rp":
                return node
            if token[0] == "op":
                self.take()
                node = Op(token[1], node, self.operand())
            else:
                node = Op("AND", node, self.operand())

    def operand(self) -> Node:
        token = self.peek()
        if token is None:
            raise QueryTranslationError("query ends where a term was expected")
        if token[0] == "lp":
            self.take()
            node = self.expression()
            if self.peek() is None:
                # PubMed closes a group the query left open at its end. Do the same.
                self.notes.append("an unclosed '(' in the query was closed at its end")
                return node
            self.take()
            return node
        leaf = self.leaf()
        if self.peek() and self.peek()[0] == "colon":
            self.take()
            return Range(leaf, self.leaf())
        return leaf

    def leaf(self) -> Leaf:
        token = self.take()
        if token[0] == "phrase":
            words, phrase, raw = token[1].split(), True, f'"{token[1]}"'
        elif token[0] == "word":
            # Bare words written together are one term. Tagged, PubMed searches them as a
            # phrase ("cue reactiv*[tiab]"); untagged, its term mapping reads them as one
            # concept ("frontotemporal dementia OR bvFTD"), so they bind tighter than the
            # operators around them, and every word must match.
            words = [token[1]]
            while self.peek() and self.peek()[0] == "word":
                words.append(self.take()[1])
            raw = " ".join(words)
            phrase = len(words) > 1 and bool(self.peek() and self.peek()[0] == "tag")
        else:
            raise QueryTranslationError(f"unexpected {token[1]!r}")
        tag = ""
        if self.peek() and self.peek()[0] == "tag":
            tag_raw = self.take()[1]
            tag = _normalize_tag(tag_raw)
            raw += tag_raw
        return Leaf(words=words, phrase=phrase, tag=tag, raw=raw)


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------

def _needs_quotes(word: str) -> bool:
    return not re.fullmatch(r"[A-Za-z0-9]+", word)


def _emit_words(words: List[str], phrase: bool) -> Tuple[str, bool]:
    """Return (NeuroStore text, approximated) for one term."""
    approximated = False
    words = [w.replace("*", "") if i < len(words) - 1 else w for i, w in enumerate(words)]
    if any("*" in w for w in words[:-1]):
        approximated = True
    last = words[-1]
    variants = [words]
    if last.endswith("*"):
        stem = last.rstrip("*")
        approximated = True
        if len(re.sub(r"[^A-Za-z]", "", stem)) >= MIN_TRUNCATION_PREFIX:
            variants = [words[:-1] + [stem + ending] for ending in TRUNCATION_ENDINGS]
        else:
            variants = [words[:-1] + [stem]]
    rendered = []
    for variant in variants:
        if phrase or (len(variant) == 1 and _needs_quotes(variant[0])):
            text = '"' + " ".join(variant).replace('"', "") + '"'
        elif len(variant) > 1:
            text = "(" + " ".join(f'"{w}"' if _needs_quotes(w) else w for w in variant) + ")"
        else:
            text = variant[0]
        rendered.append(text)
    rendered = list(dict.fromkeys(rendered))
    return (rendered[0] if len(rendered) == 1 else "(" + " OR ".join(rendered) + ")"), approximated


def _mesh_phrase(words: List[str]) -> List[str]:
    """'Magnetic Resonance Imaging, Functional' -> functional magnetic resonance imaging."""
    text = " ".join(words)
    if "," in text:
        head, _, qualifier = text.partition(",")
        text = f"{qualifier.strip()} {head.strip()}"
    return [w for w in re.split(r"[\s\-]+", text) if w]


def translate_pubmed_query(query: str) -> Translation:
    """Translate a PubMed query, reporting everything that could not be carried over."""
    parser = _Parser(_tokens(query))
    tree = parser.parse()
    result = Translation(query="", approximated=list(parser.notes))

    def visit(node: Node, in_or: bool) -> Optional[str]:
        if isinstance(node, Range):
            result.dropped.append(node.raw)
            return None
        if isinstance(node, Leaf):
            tag = node.tag
            if tag in DROP_TAGS:
                result.dropped.append(node.raw)
                return None
            if tag in MESH_TAGS:
                if not in_or:
                    result.dropped.append(node.raw)
                    return None
                result.approximated.append(node.raw)
                return _emit_words(_mesh_phrase(node.words), True)[0]
            if tag not in TEXT_TAGS:
                result.approximated.append(node.raw)
            text, approximated = _emit_words(node.words, node.phrase)
            if approximated:
                result.approximated.append(node.raw)
            return text
        left = visit(node.left, node.op == "OR")
        right = visit(node.right, node.op == "OR")
        if node.op == "NOT":
            if right is None:
                return left
            if left is None:
                result.dropped.append("a leading NOT clause")
                return None
            return f"({left} NOT {right})"
        if left is None or right is None:
            return left if right is None else right
        return f"({left} {node.op} {right})"

    negated: List[Node] = []
    node = tree
    while isinstance(node, Op) and node.op == "NOT":
        negated.insert(0, node.right)
        node = node.left
    include = visit(node, False)
    exclude = [text for text in (visit(n, False) for n in negated) if text is not None]
    if include is None:
        if negated:
            result.dropped.append("a leading NOT clause")
        raise QueryTranslationError("nothing in the query can be searched in NeuroStore")
    text = include
    for clause in exclude:
        text = f"({text} NOT {clause})"
    result.query = _strip_outer(text)
    result.include = _strip_outer(include)
    result.exclude = [_strip_outer(clause) for clause in exclude]
    return result


def _strip_outer(text: str) -> str:
    return text[1:-1] if text.startswith("(") and _balanced_outer(text) else text


def _balanced_outer(text: str) -> bool:
    depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and i < len(text) - 1:
                return False
    return depth == 0
