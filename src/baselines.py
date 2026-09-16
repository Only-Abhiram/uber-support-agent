"""
baselines.py

Two baselines required by the assignment:

  TRIVIAL  = the keyword/regex rule tagger in intents.py. Zero training,
             zero model, fully deterministic and explainable. This is
             *also* what bootstrapped the training labels for the simple
             baseline below (see DECISION_LOG.md item 3 on why we didn't
             just hand-label 40k examples).

  SIMPLE   = TF-IDF (word 1-2 grams) + multinomial Logistic Regression,
             trained on the FULL corpus using the trivial rule tagger's
             output as weak/pseudo labels (with golden-set conversation_ids
             excluded from training to avoid leakage into the eval set).
             This is the standard "before you reach for an LLM" baseline:
             cheap, fast, no API calls, and a fair test of whether the
             extra intent signal in the text is even linearly separable
             from TF-IDF features.

Both are scored against eval/golden_set.jsonl, which has HAND-corrected
intent labels (see eval/build_golden_set.py) -- so this eval also reports,
honestly, how often the simple baseline just reproduces the trivial
baseline's mistakes (it was trained on them).
"""
import json
import sys
from collections import Counter

sys.path.insert(0, "src")
from intents import rule_based_intent, PRIORITY  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib


def load_cases(path="data/cases.jsonl"):
    return [json.loads(l) for l in open(path)]


def load_golden(path="eval/golden_set.jsonl"):
    return [json.loads(l) for l in open(path)]


def train_simple_baseline(cases, golden_conv_ids, model_out="eval/results/simple_baseline.joblib"):
    train_cases = [c for c in cases if c["conversation_id"] not in golden_conv_ids]
    texts = [c["customer_text_norm"] for c in train_cases]
    labels = [rule_based_intent(c["customer_text_clean"]) for c in train_cases]

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
    clf.fit(X, labels)
    joblib.dump({"vectorizer": vec, "clf": clf}, model_out)
    print(f"Trained simple baseline on {len(train_cases)} examples (excluded {len(cases)-len(train_cases)} golden-set rows)")
    return vec, clf


def normalize_for_matching(text):
    # local import to avoid circulars in quick scripts
    from data_prep import normalize_for_matching as f
    return f(text)


def eval_trivial(golden):
    y_true = [g["gold_intent"] for g in golden]
    y_pred = [rule_based_intent(g["text"]) for g in golden]
    return y_true, y_pred


def eval_simple(golden, vec, clf):
    from data_prep import normalize_for_matching
    y_true = [g["gold_intent"] for g in golden]
    X = vec.transform([normalize_for_matching(g["text"]) for g in golden])
    y_pred = list(clf.predict(X))
    return y_true, y_pred


def report(name, y_true, y_pred, out_path):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    rep = classification_report(y_true, y_pred, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"accuracy={acc:.3f}  macro_f1={f1_macro:.3f}  weighted_f1={f1_weighted:.3f}")
    with open(out_path, "w") as f:
        f.write(f"{name}\naccuracy={acc:.4f}\nmacro_f1={f1_macro:.4f}\nweighted_f1={f1_weighted:.4f}\n\n")
        f.write(rep)
    return {"name": name, "accuracy": acc, "macro_f1": f1_macro, "weighted_f1": f1_weighted}


if __name__ == "__main__":
    cases = load_cases()
    golden = load_golden()
    golden_conv_ids = {g["conversation_id"] for g in golden}

    yt, yp = eval_trivial(golden)
    trivial_metrics = report("TRIVIAL (rule-based)", yt, yp, "eval/results/trivial_report.txt")

    vec, clf = train_simple_baseline(cases, golden_conv_ids)
    yt2, yp2 = eval_simple(golden, vec, clf)
    simple_metrics = report("SIMPLE (TF-IDF + LogisticRegression)", yt2, yp2, "eval/results/simple_report.txt")

    with open("eval/results/baseline_summary.json", "w") as f:
        json.dump({"trivial": trivial_metrics, "simple": simple_metrics}, f, indent=2)
    print("\nWrote eval/results/baseline_summary.json")
