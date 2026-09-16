#!/usr/bin/env bash
# Reproduces the headline (non-LLM) results in well under 15 minutes.
# Usage: RAW_DATA=/path/to/uber_conversations.jsonl ./run_pipeline.sh
set -euo pipefail

RAW_DATA="${RAW_DATA:-data/raw/uber_conversations.jsonl}"

if [ ! -f "$RAW_DATA" ]; then
  echo "Raw data not found at $RAW_DATA"
  echo "Download 'Customer Support on Twitter' from Kaggle, filter/group to Uber"
  echo "conversations (or point RAW_DATA at an already-grouped JSONL -- see"
  echo "README.md 'Data format' section for the expected schema), and retry."
  exit 1
fi

echo "== 1/4 Data prep =="
python3 src/data_prep.py "$RAW_DATA" data/cases.jsonl

echo "== 2/4 Fit retriever (local, no API calls) =="
python3 src/retrieval.py

echo "== 3/4 Baselines: trivial rule tagger + TF-IDF/LogReg =="
python3 src/baselines.py

echo "== 4/4 Build golden set from candidates (already hand-labeled; see eval/build_golden_set.py) =="
(cd eval && python3 build_golden_set.py)

echo
echo "Done. Baseline metrics: eval/results/baseline_summary.json"
echo
echo "To also run the Gemini-backed agent + LLM-judge (requires GEMINI_API_KEY"
echo "and network access to generativelanguage.googleapis.com):"
echo "  export GEMINI_API_KEY=..."
echo "  python3 src/eval_harness.py --limit 40"
