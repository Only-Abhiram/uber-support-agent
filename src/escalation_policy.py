"""
escalation_policy.py

Defines when the agent should AUTO_HANDLE a message (send an on-brand,
templated triage/acknowledgement reply itself) vs ESCALATE it to a human
agent, and states the reason.

Framing (see REPORT.md "Problem framing"): because this dataset only
contains the PUBLIC, first-touch tweet exchange, we deliberately do NOT
pretend the agent can independently resolve refunds, account restores, or
fare adjustments — those require verified account/trip data the agent
doesn't have here. "Auto-handle" therefore means: the agent is confident
enough in the intent to send the correct on-brand acknowledgement +
next-step (e.g. "please DM your trip ID so we can look into the fare") from
its own drafting, without needing a human to pick the template or judge the
tone. "Escalate" means a human should see it before anything is sent,
because getting it wrong is costly (safety, legal, PR, a vulnerable
customer, or the model's own low confidence).

Signals, in order of precedence:
  1. Hard safety/legal/vulnerability triggers -> always escalate,
     regardless of predicted intent or model confidence.
  2. Intent-level default risk tier (set in intents.py).
  3. Classifier confidence: below threshold -> escalate as "unsure of intent".
  4. Otherwise: low/medium risk + confident -> auto-handle.
"""
import re

HARD_ESCALATE_PATTERNS = [
    (r"\b(kill|rape|assault(ed)?|weapon|gun|knife)\b", "possible violent/criminal safety incident"),
    (r"\b(su[ie]cid|self[- ]harm|kill myself)\b", "possible self-harm risk"),
    (r"\b(lawyer|lawsuit|su[ei]ng|legal action|attorney|police report)\b", "legal threat / police involvement"),
    (r"\b(minor|child|under ?age|my (son|daughter) (was|is) \d)\b", "possible minor involved"),
    (r"\b(disab(led|ility)|wheelchair|ada\b|service (dog|animal))\b", "accessibility / ADA-sensitive case"),
    (r"\b(journalist|reporter|news ?(story|outlet)|press|going viral)\b", "PR / media risk"),
    (r"\bdiscriminat|racist|racism|sexist\b", "discrimination allegation"),
]
_COMPILED_HARD = [(re.compile(p, re.IGNORECASE), reason) for p, reason in HARD_ESCALATE_PATTERNS]

CONFIDENCE_THRESHOLD = 0.55


def decide(text: str, predicted_intent: str, intent_default_risk: str, confidence: float):
    """Returns (decision, reason) where decision in {"AUTO_HANDLE", "ESCALATE"}."""
    for pat, reason in _COMPILED_HARD:
        if pat.search(text):
            return "ESCALATE", f"hard trigger: {reason}"

    if confidence < CONFIDENCE_THRESHOLD:
        return "ESCALATE", f"low classifier confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD})"

    if intent_default_risk == "high":
        return "ESCALATE", "intent is high-risk by default (safety/account-security category)"

    if intent_default_risk == "medium":
        # medium-risk intents are auto-handled only when the model is quite sure
        if confidence >= 0.75:
            return "AUTO_HANDLE", "medium-risk intent but high confidence and no hard triggers"
        return "ESCALATE", f"medium-risk intent with only moderate confidence ({confidence:.2f})"

    return "AUTO_HANDLE", "low-risk intent, no hard triggers, confidence sufficient"
