"""
data_prep.py
Loads the raw Uber conversation JSONL (already grouped/threaded, one file per
conversation with `messages: [...]`), and produces a flat, analysis-ready
table of (customer_query, brand_first_reply, conversation-level metadata).

Design notes (see DECISION_LOG.md for the "why"):
- We anchor on the FIRST customer message and the FIRST brand reply in each
  thread. This is the "triage" turn: it's the one a real-time agent (human
  or AI) would actually be facing when a new ticket lands. Later turns
  depend on private DM content we don't have, so we don't model them.
- We keep the full message list too, in case later analysis needs it
  (e.g. to check how many threads clearly escalate to DM/private channel).
- We do light text cleaning (strip @handles used only for threading, strip
  URLs into a placeholder) but we DO NOT over-clean: casing, punctuation and
  emoji carry sentiment/urgency signal we want the classifier and the
  escalation policy to see.
"""
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

URL_RE = re.compile(r"https?://\S+")
HANDLE_RE = re.compile(r"@\w+")


def strip_leading_handles(text: str) -> str:
    """Remove leading @handle mentions (Twitter reply artifacts) but keep
    any @mentions that occur mid-sentence (those are often meaningful,
    e.g. tagging a second airline)."""
    return re.sub(r"^(@\w+\s*)+", "", text).strip()


def normalize_for_matching(text: str) -> str:
    """Aggressive normalization used ONLY for clustering / TF-IDF, never
    shown to the user or the LLM."""
    t = text.lower()
    t = URL_RE.sub(" <url> ", t)
    t = HANDLE_RE.sub(" ", t)
    t = re.sub(r"[^a-z0-9<>#'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class Case:
    conversation_id: str
    message_count: int
    customer_text_raw: str
    customer_text_clean: str
    customer_text_norm: str
    brand_reply_raw: Optional[str]
    brand_reply_clean: Optional[str]
    n_customer_turns: int
    n_brand_turns: int
    created_at: Optional[str]


def load_cases(path: str, limit: Optional[int] = None) -> list:
    cases = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            d = json.loads(line)
            msgs = d["messages"]
            first_customer = next((m for m in msgs if m["speaker"] == "customer"), None)
            first_brand = next((m for m in msgs if m["speaker"] == "uber"), None)
            if not first_customer:
                continue
            n_cust = sum(1 for m in msgs if m["speaker"] == "customer")
            n_brand = sum(1 for m in msgs if m["speaker"] == "uber")
            raw = first_customer["text"]
            clean = strip_leading_handles(raw)
            norm = normalize_for_matching(clean)
            brand_raw = first_brand["text"] if first_brand else None
            brand_clean = strip_leading_handles(brand_raw) if brand_raw else None
            cases.append(Case(
                conversation_id=d["conversation_id"],
                message_count=d["message_count"],
                customer_text_raw=raw,
                customer_text_clean=clean,
                customer_text_norm=norm,
                brand_reply_raw=brand_raw,
                brand_reply_clean=brand_clean,
                n_customer_turns=n_cust,
                n_brand_turns=n_brand,
                created_at=first_customer.get("created_at"),
            ))
    return cases


def cases_to_jsonl(cases, path):
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(asdict(c)) + "\n")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/uber_conversations.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/cases.jsonl"
    cases = load_cases(src)
    cases_to_jsonl(cases, out)
    print(f"Loaded {len(cases)} cases -> {out}")
