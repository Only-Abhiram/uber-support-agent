"""
llm_judge.py

LLM-as-judge for the AGENT'S DRAFTED REPLY (not the intent label). Scores
each (customer_message, intent, draft_reply) on a 4-dimension rubric,
1-5 each, plus an overall verdict. This is a SEPARATE Gemini call from
drafting -- the judge never sees which retrieval examples were used, only
the final customer message + reply, so it can't just check "does this
copy the retrieved template" (that would be grading the retrieval, not
the reply).

Rubric dimensions (each 1-5):
  - relevance:    does the reply actually address what THIS customer said,
                   not a generic template unrelated to their specifics?
  - faithfulness: does the reply avoid inventing facts (refund amounts,
                   promised timelines, causes) not present in the message?
                   This is the most important dimension given the dataset's
                   deflection-heavy ground truth (see REPORT.md) -- a reply
                   that promises "we've refunded your $20" it can't know is
                   worse than a vague-but-honest one.
  - tone:         warm, professional, appropriately serious for the issue
                   (a safety complaint should not get a cheerful reply).
  - actionability: does it give the customer a concrete, sane next step?

We separately validate this judge against human labels (see
eval/judge_agreement.py) rather than trusting it blindly -- an LLM judge
that hasn't been checked against a human is exactly the kind of unverified
metric the assignment's "what's misleading about your headline number"
section calls out.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from agent import _call_gemini  # noqa

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "integer"},
        "faithfulness": {"type": "integer"},
        "tone": {"type": "integer"},
        "actionability": {"type": "integer"},
        "overall_comment": {"type": "string"},
    },
    "required": ["relevance", "faithfulness", "tone", "actionability", "overall_comment"],
}

JUDGE_SYSTEM = """You are grading a customer support agent's draft reply to a
tweet, for a company auditing its AI support system. Score each dimension
1 (bad) to 5 (excellent):

relevance: does the reply engage with the SPECIFICS of this customer's message?
faithfulness: does the reply AVOID inventing facts/amounts/promises not in
  the customer's message (a vague-but-honest reply should score HIGH here;
  a specific-but-fabricated promise should score LOW even if it sounds nice)?
tone: is the tone warm, professional, and appropriately serious for the issue?
actionability: does the customer know what to do next after reading this?

Be strict on faithfulness in particular -- do not reward confident-sounding
fabrication. Return ONLY the JSON object."""


def judge_reply(customer_text: str, intent: str, reply: str) -> dict:
    prompt = f'Customer message ({intent}): "{customer_text}"\n\nAgent\'s draft reply: "{reply}"'
    return _call_gemini(JUDGE_SYSTEM, prompt, JUDGE_SCHEMA)


if __name__ == "__main__":
    demo = judge_reply(
        "your driver charged me twice for the same ride and support won't answer",
        "BILLING_FARE_DISPUTE",
        "So sorry about that! Please send us a DM with your trip ID and we'll take a look at the duplicate charge.",
    )
    print(json.dumps(demo, indent=2))
