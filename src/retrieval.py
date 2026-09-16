"""
retrieval.py

Retrieval-for-grounding. Given a new customer message and its predicted
intent, find the K most similar HISTORICAL customer messages of the same
intent that got a brand reply, and return those (query, reply) pairs as
grounding context for the LLM's draft.

Deliberate choice: retrieval itself does NOT call any LLM or embedding API.
It's plain TF-IDF cosine similarity, computed locally. Only the final
drafting step calls Gemini. This keeps the expensive/rate-limited part of
the pipeline (LLM calls) to exactly one call per message for classification
and one for drafting, and keeps retrieval fully reproducible/offline (see
DECISION_LOG.md item 4).

Important honesty note (goes in REPORT.md): because this dataset is the
PUBLIC Twitter reply only, the "historical resolution" we retrieve is
almost always a triage/deflection template ("send us a DM..."), not an
actual resolution (refund issued, account restored, etc. -- that happened
in a private DM channel this dataset doesn't contain). So "grounded in how
the brand has historically resolved similar issues" here means "grounded in
the brand's actual, historical FIRST-TOUCH voice and triage pattern for
this intent" -- which is exactly what the agent is responsible for at this
stage of the conversation. See REPORT.md "Problem framing".
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from intents import rule_based_intent  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


class IntentRetriever:
    def __init__(self):
        self.by_intent = defaultdict(list)     # intent -> list of case dicts
        self.vec_by_intent = {}
        self.X_by_intent = {}

    def fit(self, cases):
        for c in cases:
            if not c.get("brand_reply_clean"):
                continue
            intent = rule_based_intent(c["customer_text_clean"])
            self.by_intent[intent].append(c)
        for intent, group in self.by_intent.items():
            texts = [g["customer_text_norm"] for g in group]
            vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)
            X = vec.fit_transform(texts)
            self.vec_by_intent[intent] = vec
            self.X_by_intent[intent] = X
        return self

    def save(self, path="eval/results/retriever.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="eval/results/retriever.joblib"):
        return joblib.load(path)

    def retrieve(self, query_text_norm: str, intent: str, k: int = 3):
        if intent not in self.vec_by_intent or self.X_by_intent[intent].shape[0] == 0:
            # fall back to a small cross-intent pool so we always return *something*
            intent = max(self.by_intent, key=lambda k_: len(self.by_intent[k_]))
        vec = self.vec_by_intent[intent]
        X = self.X_by_intent[intent]
        qv = vec.transform([query_text_norm])
        sims = cosine_similarity(qv, X).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        group = self.by_intent[intent]
        out = []
        for i in top_idx:
            g = group[i]
            out.append({
                "similarity": float(sims[i]),
                "historical_customer_msg": g["customer_text_clean"],
                "historical_brand_reply": g["brand_reply_clean"],
            })
        return out


if __name__ == "__main__":
    from data_prep import normalize_for_matching
    cases = [json.loads(l) for l in open("data/cases.jsonl")]
    r = IntentRetriever().fit(cases)
    r.save()
    print("Fitted retriever over intents:", {k: len(v) for k, v in r.by_intent.items()})

    # smoke test
    demo = "your driver charged me twice for the same ride and support won't answer"
    intent = rule_based_intent(demo)
    hits = r.retrieve(normalize_for_matching(demo), intent, k=3)
    print(f"\nDemo query -> intent={intent}")
    for h in hits:
        print(f"  sim={h['similarity']:.2f}  cust: {h['historical_customer_msg'][:80]}")
        print(f"           reply: {h['historical_brand_reply'][:80]}")
