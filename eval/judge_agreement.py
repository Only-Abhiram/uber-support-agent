"""
judge_agreement.py

Required deliverable: "evidence of how well your judge agrees with a human."

Step 1 (this script, `sample`): pull N=30 rows from eval/results/agent_eval.jsonl
(produced by eval_harness.py) into eval/results/judge_agreement_sheet.csv,
with the judge's 4 scores already filled in and four blank human_* columns.

Step 2 (manual, YOU do this): open the CSV, read each customer message +
draft reply yourself, and fill in human_relevance / human_faithfulness /
human_tone / human_actionability (1-5 each) WITHOUT looking at the judge's
scores first (cover them / sort by a random column while rating, so you
aren't anchored). This is the actual human-labeling step -- there is no way
to automate genuine human judgment, and reviewers will ask you to explain
your ratings live, so do this pass yourself.

Step 3 (this script, `score`): once the CSV is filled in, computes
Spearman correlation and quadratic-weighted Cohen's kappa (scores treated
as ordinal) between judge and human, per dimension, plus mean absolute
difference. Low agreement on a dimension means: don't trust that dimension
of the judge's score in the report without a human spot-check, and say so
explicitly in REPORT.md's "what's misleading" section.
"""
import argparse
import csv
import json
import random

DIMS = ["relevance", "faithfulness", "tone", "actionability"]


def cmd_sample(args):
    rows = [json.loads(l) for l in open(args.eval_file) if not json.loads(l).get("error")]
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "customer_text", "intent", "draft_reply"]
                    + [f"judge_{d}" for d in DIMS] + [f"human_{d}" for d in DIMS])
        for r in sample:
            w.writerow([r["id"], r["input"], r["intent"], r["draft_reply"]]
                        + [r["judge"][d] for d in DIMS] + [""] * len(DIMS))
    print(f"Wrote {len(sample)} rows to {args.out}. Fill in the human_* columns, "
          f"then run: python judge_agreement.py score --sheet {args.out}")


def cmd_score(args):
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    rows = list(csv.DictReader(open(args.sheet)))
    rows = [r for r in rows if all(r[f"human_{d}"].strip() for d in DIMS)]
    if len(rows) < 5:
        print(f"Only {len(rows)} rows have human ratings filled in -- fill in more "
              f"of {args.sheet} before scoring (need at least ~15-20 for a meaningful check).")
        return
    print(f"Scoring agreement on {len(rows)} rated rows.\n")
    for d in DIMS:
        judge_vals = [int(r[f"judge_{d}"]) for r in rows]
        human_vals = [int(r[f"human_{d}"]) for r in rows]
        rho, _ = spearmanr(judge_vals, human_vals)
        kappa = cohen_kappa_score(judge_vals, human_vals, weights="quadratic")
        mad = sum(abs(j - h) for j, h in zip(judge_vals, human_vals)) / len(rows)
        print(f"{d:16s} spearman_rho={rho:.2f}  quadratic_kappa={kappa:.2f}  mean_abs_diff={mad:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sample")
    s1.add_argument("--eval-file", dest="eval_file", default="eval/results/agent_eval.jsonl")
    s1.add_argument("--out", default="eval/results/judge_agreement_sheet.csv")
    s1.add_argument("--n", type=int, default=30)
    s1.add_argument("--seed", type=int, default=0)
    s1.set_defaults(func=cmd_sample)

    s2 = sub.add_parser("score")
    s2.add_argument("--sheet", default="eval/results/judge_agreement_sheet.csv")
    s2.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)
