"""
eval_harness.py

Runs the full agent (Gemini classify -> retrieve -> Gemini draft -> policy
decide) over N examples from eval/golden_set.jsonl, and:

  1. Scores intent classification accuracy/macro-F1 against gold_intent,
     for direct comparison with the trivial and simple baselines
     (src/baselines.py).
  2. Scores the escalation decision against gold_escalate, reporting
     precision/recall FOR THE ESCALATE CLASS specifically (a missed
     safety escalation is much worse than an unnecessary one, so overall
     accuracy alone would hide the failure mode that matters most).
  3. Runs the LLM-judge (src/llm_judge.py) on every drafted reply and
     reports mean scores per rubric dimension.

Usage:
    export GEMINI_API_KEY=...
    python src/eval_harness.py --limit 10          # quick repro
    python src/eval_harness.py --limit 204         # full golden set

Rate limiting: examples are run ONE AT A TIME (not concurrently), with a
fixed --interval pause between each. Each example makes 3 Gemini calls
(classify, draft, judge), so raise --interval if you still see 429s.

Output: as each example finishes, this prints the customer query and the
agent's full action (intent, confidence, decision, reply) to the CLI, AND
appends the same thing to eval/results/agent_eval_log.txt -- so you have
a readable transcript, not just a JSON blob, and it's written incrementally
(not just at the end) in case a later example errors or you Ctrl-C out.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from agent import UberSupportAgent  # noqa
from retrieval import IntentRetriever  # noqa
from llm_judge import judge_reply  # noqa
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def run_one(agent, row):
    try:
        result = agent.handle(row["text"])
        judged = judge_reply(row["text"], result["intent"], result["draft_reply"])
        result.update({
            "id": row["id"],
            "gold_intent": row["gold_intent"],
            "gold_escalate": row["gold_escalate"],
            "judge": judged,
            "error": None,
        })
    except Exception as e:  # noqa
        result = {"id": row["id"], "gold_intent": row["gold_intent"],
                  "gold_escalate": row["gold_escalate"], "error": str(e)}
    return result


def format_transcript_entry(i, total, r):
    """Human-readable block for one example: the query in, the action taken."""
    lines = [f"[{i}/{total}] id={r['id']}"]
    if r.get("error"):
        lines.append(f"  ERROR: {r['error']}")
        return "\n".join(lines)
    lines.append(f'  QUERY:     "{r["input"]}"')
    lines.append(f"  INTENT:    {r['intent']}  (confidence={r['intent_confidence']:.2f}, gold={r['gold_intent']})")
    lines.append(f"  DECISION:  {r['decision']}  -- {r['decision_reason']}  (gold_escalate={r['gold_escalate']})")
    lines.append(f'  REPLY:     "{r["draft_reply"]}"')
    if r.get("judge"):
        j = r["judge"]
        lines.append(f"  JUDGE:     relevance={j['relevance']} faithfulness={j['faithfulness']} "
                      f"tone={j['tone']} actionability={j['actionability']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--interval", type=float, default=8.0,
                     help="seconds to wait between each example (avoids rate limits)")
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--out", default="eval/results/agent_eval.jsonl")
    ap.add_argument("--log", default="eval/results/agent_eval_log.txt",
                     help="human-readable transcript of query -> action per example")
    args = ap.parse_args()

    golden = [json.loads(l) for l in open(args.golden)][: args.limit]
    retriever = IntentRetriever.load("eval/results/retriever.joblib")
    agent = UberSupportAgent(retriever)

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_f = open(args.log, "w")

    results = []
    for i, row in enumerate(golden):
        r = run_one(agent, row)
        results.append(r)

        entry = format_transcript_entry(i + 1, len(golden), r)
        print(entry)
        print("-" * 80)
        log_f.write(entry + "\n" + ("-" * 80) + "\n")
        log_f.flush()  # so the file is readable even if the run gets interrupted

        if i < len(golden) - 1:
            time.sleep(args.interval)

    log_f.close()

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if not r.get("error")]
    errored = len(results) - len(ok)
    if errored:
        print(f"\n{errored}/{len(results)} examples errored (see error field) -- "
              f"excluded from metrics below.")

    y_true = [r["gold_intent"] for r in ok]
    y_pred = [r["intent"] for r in ok]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    et = [r["gold_escalate"] for r in ok]
    ep = [r["decision"] == "ESCALATE" for r in ok]
    esc_precision = precision_score(et, ep, zero_division=0)
    esc_recall = recall_score(et, ep, zero_division=0)

    dims = ["relevance", "faithfulness", "tone", "actionability"]
    judge_means = {d: sum(r["judge"][d] for r in ok) / len(ok) for d in dims} if ok else {}

    summary = {
        "n": len(ok),
        "n_errored": errored,
        "intent_accuracy": acc,
        "intent_macro_f1": macro_f1,
        "escalation_precision": esc_precision,
        "escalation_recall": esc_recall,
        "judge_mean_scores": judge_means,
    }
    print("\n=== AGENT (Gemini) SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("eval/results/agent_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}, {args.log}, and eval/results/agent_summary.json")


if __name__ == "__main__":
    main()