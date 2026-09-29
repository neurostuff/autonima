#!/usr/bin/env bash
# Run each pending batch of one stage in a fresh, headless agent session.
#
# Use this when the harness cannot spawn subagents in-session, or to run a large
# stage outside an interactive session. Each batch gets a clean context, which is
# the point: the 40th batch is judged with the same instructions as the first.
#
#   AGENT_CMD='claude -p --allowedTools Read,Write,Glob' \
#     scripts/run_batches.sh REVIEW abstract [PARALLEL]
#   AGENT_CMD='codex exec --full-auto' \
#     scripts/run_batches.sh REVIEW fulltext 2
#
# AGENT_CMD is any command that takes a prompt as its final argument, runs to
# completion, and can read and write files. Check your harness's flags for
# non-interactive runs and file permissions; the ones above are examples.
# Afterwards run `ledger.py ingest` as usual.
set -euo pipefail

review="${1:?usage: run_batches.sh REVIEW STAGE [PARALLEL]}"
stage="${2:?usage: run_batches.sh REVIEW STAGE [PARALLEL]}"
parallel="${3:-4}"
: "${AGENT_CMD:?set AGENT_CMD, e.g. AGENT_CMD='claude -p --allowedTools Read,Write,Glob'}"

skills_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(cd "$review" && pwd)/work/$stage"

run_one() {
  batch="$1"
  skill="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["skill"])' "$batch")"
  out="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$batch")"
  if [ -e "$out" ]; then
    echo "skip $(basename "$batch") (output exists)"
    return 0
  fi
  prompt="Read ${skills_dir}/${skill}/SKILL.md and follow it exactly. Your batch file is ${batch}. Process every item in it and write your output to the path in the batch's \"output\" field. Do not read or write any other part of the review folder except the files the batch names. When done, reply with only the number of items you wrote."
  log="${batch%.json}.agent.log"
  # shellcheck disable=SC2086
  if $AGENT_CMD "$prompt" >"$log" 2>&1; then
    echo "done $(basename "$batch")"
  else
    echo "FAILED $(basename "$batch") (see $log); it stays pending"
  fi
}
export -f run_one
export skills_dir AGENT_CMD

find "$work" -maxdepth 1 -name 'batch_*.json' | sort | xargs -r -P "$parallel" -I{} bash -c 'run_one "$@"' _ {}
echo "now run: python ${skills_dir}/cbma-review/scripts/ledger.py ingest ${review} --stage ${stage} --agent '<harness>/<model>'"
