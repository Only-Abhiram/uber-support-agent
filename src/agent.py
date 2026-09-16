"""
agent.py

The actual "AI support agent" for Uber, wired to the Gemini API.

Pipeline for one incoming message:
  1. classify_intent()  -> LLM call #1. Zero/few-shot classification over
     the fixed taxonomy in intents.py, forced into structured JSON output
     (intent, confidence 0-1, one-line rationale). We ask the LLM (not a
     softmax) for confidence because we have no local model probabilities
     here; see DECISION_LOG.md item 7 for why we treat this as a rough,
     self-reported signal rather than a calibrated probability, and how
     the eval harness sanity-checks it.
  2. retrieve()         -> no LLM call. Local TF-IDF nearest neighbors
     within the predicted intent (src/retrieval.py).
  3. draft_reply()      -> LLM call #2. Grounded generation: the retrieved
     historical (customer, brand reply) pairs are given as few-shot
     examples of the brand's voice/pattern for this intent, NOT as facts
     to copy verbatim. The prompt explicitly tells the model these are
     tone/pattern references, and forbids inventing specific facts
     (refund amounts, timelines, names) that weren't in the retrieved
     examples or the incoming message.
  4. decide()           -> no LLM call. escalation_policy.py combines the
     hard-trigger regexes, the intent's default risk tier, and the
     confidence from step 1.

Only 2 LLM calls per message. Both use Gemini's structured JSON output mode
so results are parseable without regex-scraping model prose.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
from intents import INTENTS, INTENTS_BY_KEY  # noqa
from escalation_policy import decide as escalation_decide  # noqa
from data_prep import normalize_for_matching, strip_leading_handles  # noqa
from retrieval import IntentRetriever  # noqa

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

INTENT_TAXONOMY_TEXT = "\n".join(
    f"- {i.key}: {i.description}" for i in INTENTS
)
VALID_KEYS = [i.key for i in INTENTS]


def _call_gemini(system_instruction: str, user_text: str, response_schema: dict,
                  max_retries: int = 3, timeout: int = 30) -> dict:
    """Minimal dependency-free REST call to Gemini generateContent with
    structured JSON output. Retries on 429/5xx with backoff."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it as an environment variable "
            "(see .env.example) -- never hardcode it in source or commit it."
        )
    body = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "temperature": 0.2,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
        method="POST",
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 and attempt < max_retries - 1:
                # Respect Retry-After if Gemini sends one, otherwise back off hard
                # (free tier is ~10-15 requests/min, so 1/2/4s isn't enough).
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (5 * (2 ** attempt))
                print(f"  [rate limited, waiting {wait:.0f}s before retry {attempt+1}/{max_retries}]")
                time.sleep(wait)
                continue
            if e.code in (500, 502, 503) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode()[:500]}") from e
        except (TimeoutError, urllib.error.URLError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API timed out after {timeout}s (attempt {attempt+1}/{max_retries})") from e
    raise last_err


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": VALID_KEYS},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["intent", "confidence", "rationale"],
}

CLASSIFY_SYSTEM = f"""You are an intent classifier for Uber customer support tweets.
Classify the customer's message into EXACTLY ONE of these intents:

{INTENT_TAXONOMY_TEXT}

Rules:
- Pick the single best-fitting intent. If a message mentions a safety, harassment,
  or discrimination concern, prefer DRIVER_SAFETY_BEHAVIOR or ACCOUNT_ACCESS_SECURITY
  over a more generic bucket, even if it also contains a billing detail.
- "confidence" is your OWN calibrated estimate (0.0-1.0) that this is the correct
  label, not a formality -- use lower values for genuinely ambiguous messages.
- "rationale" is one short sentence.
Return ONLY the JSON object."""


def classify_intent(text: str) -> dict:
    return _call_gemini(CLASSIFY_SYSTEM, text, CLASSIFY_SCHEMA)


DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
    },
    "required": ["reply"],
}


def build_draft_prompt(customer_text: str, intent_key: str, retrieved: list) -> str:
    intent = INTENTS_BY_KEY[intent_key]
    examples_block = "\n".join(
        f'  Customer: "{r["historical_customer_msg"]}"\n  Uber replied: "{r["historical_brand_reply"]}"'
        for r in retrieved
    ) or "  (no close historical examples found -- use general brand voice)"
    return f"""Intent: {intent.name} -- {intent.description}

Here are real historical examples of how Uber's support account has replied to
SIMILAR customer messages (use these ONLY to match Uber's tone, structure, and
what it typically asks for -- do NOT copy specific numbers, names, or claims
from them into your reply):

{examples_block}

New customer message to reply to:
"{customer_text}"

Write Uber's public reply to this tweet."""


DRAFT_SYSTEM = """You are drafting Uber's public Twitter support reply.
Voice: warm, brief (1-2 sentences, under 250 characters), acknowledges the
specific issue in the customer's own words, and gives a concrete next step.
Because this is the public reply (not a private DM), NEVER invent or promise
a specific refund amount, compensation, timeline, or outcome you cannot
verify from the message alone -- the standard pattern is to acknowledge and
direct the customer to a verifiable next step (e.g. "please send us a DM
with your trip details so we can look into this"). If the message describes
a safety, harassment, or discrimination incident, respond with empathy and
direct them to a safety-specific channel; do not minimize it or add a
generic apology-and-DM without acknowledging the severity.
Return ONLY the JSON object with a "reply" field."""


def draft_reply(customer_text: str, intent_key: str, retrieved: list) -> str:
    prompt = build_draft_prompt(customer_text, intent_key, retrieved)
    result = _call_gemini(DRAFT_SYSTEM, prompt, DRAFT_SCHEMA)
    return result["reply"]


class UberSupportAgent:
    def __init__(self, retriever: IntentRetriever):
        self.retriever = retriever

    def handle(self, raw_customer_text: str) -> dict:
        text = strip_leading_handles(raw_customer_text)
        norm = normalize_for_matching(text)

        cls = classify_intent(text)
        intent_key = cls["intent"]
        confidence = float(cls["confidence"])

        retrieved = self.retriever.retrieve(norm, intent_key, k=3)
        reply = draft_reply(text, intent_key, retrieved)

        default_risk = INTENTS_BY_KEY[intent_key].default_risk
        decision, reason = escalation_decide(text, intent_key, default_risk, confidence)

        return {
            "input": raw_customer_text,
            "intent": intent_key,
            "intent_confidence": confidence,
            "intent_rationale": cls["rationale"],
            "draft_reply": reply,
            "decision": decision,
            "decision_reason": reason,
            "grounding_examples_used": len(retrieved),
        }


if __name__ == "__main__":
    retriever = IntentRetriever.load("eval/results/retriever.joblib")
    agent = UberSupportAgent(retriever)
    demo_messages = [
        "your driver charged me twice for the same ride and support won't answer",
        "I left my phone in the uber, please help me get it back",
        "a driver just threatened me after I asked him to slow down, I'm shaking",
    ]
    for m in demo_messages:
        print("=" * 80)
        print("IN: ", m)
        result = agent.handle(m)
        print(json.dumps(result, indent=2))