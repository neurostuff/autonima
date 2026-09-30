"""Versions for cache schemas and model-facing prompt implementations."""

CACHE_SCHEMA_VERSION = 2
ABSTRACT_SCREENING_PROMPT_VERSION = "2026-08-14.cache-v2"
FULLTEXT_SCREENING_PROMPT_VERSION = "2026-08-14.cache-v2"
ANNOTATION_PROMPT_VERSION = "2026-08-14.cache-v2"

# Folded into cache signatures only for studies read from a document source, so bumping it
# re-screens and re-annotates document runs and leaves article-run caches untouched.
DOCUMENT_PROMPT_VERSION = "2026-09-28.documents-v1"
