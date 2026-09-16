"""
intents.py

The intent taxonomy below was NOT hand-invented first and forced onto the
data. It came from unsupervised discovery:

  1. TF-IDF (1-2 grams) over the first customer message of every thread.
  2. TruncatedSVD to 100 components (LSA) to denoise the sparse TF-IDF space.
  3. KMeans, sweeping k=8..24, model-selected with silhouette score on the
     SVD space (all scores were low, ~0.02-0.04 -- expected for short, noisy,
     multi-topic tweets; see REPORT.md "what's misleading" section for why
     we didn't chase a higher silhouette by over-splitting).
  4. k=14 was inspected by hand: top TF-IDF terms per cluster + 4 random raw
     examples per cluster (see notebooks/cluster_dump.txt).

Several of the 14 clusters were clearly the same underlying intent split by
a surface n-gram (e.g. "charged twice" vs "cancellation fee" both being
"they charged me and I don't think they should have"), and one cluster
(~32% of the sample) was a catch-all of generic complaints with no coherent
top terms. So the 14 raw clusters were merged/split by hand into the 12
below -- clustering told us the *shape* of the space, we made the final
taxonomy decisions. This hybrid (unsupervised discovery -> human-curated
taxonomy) is a deliberate choice; see DECISION_LOG.md item 2.

Each intent has:
  - a short human name
  - a description (used verbatim in LLM prompts)
  - a list of regex/keyword rules used for weak/bootstrap labeling
    (the "trivial baseline" classifier IS this rule set)
  - a default risk tier used by the escalation policy
"""
from dataclasses import dataclass
import re


@dataclass
class Intent:
    key: str
    name: str
    description: str
    patterns: list          # compiled regexes, any match -> candidate label
    default_risk: str        # "low" | "medium" | "high"


INTENTS = [
    Intent(
        key="ACCOUNT_ACCESS_SECURITY",
        name="Account Access & Security",
        description="Customer's account is locked, disabled, hacked, or they suspect unauthorized access / unrecognized charges from account takeover.",
        patterns=[
            r"\bhack(ed|ing)?\b", r"account.{0,15}(disabled|locked|blocked|suspend|deactivat)",
            r"(disabled|locked|blocked|suspend).{0,15}account",
            r"unauthori[sz]ed", r"someone (else )?(used|access|took over) my account",
            r"can'?t log ?in", r"reset my password",
        ],
        default_risk="high",
    ),
    Intent(
        key="BILLING_FARE_DISPUTE",
        name="Billing / Fare Dispute",
        description="Customer was charged incorrectly: wrong fare amount, charged twice, charged after trip ended, surprise fee not related to cancellation.",
        patterns=[
            r"charg(ed|ing).{0,25}(twice|double|again|wrong|extra)",
            r"overcharg", r"wrong (amount|fare|price|charge)",
            r"refund", r"double charg", r"charged me \$?\d",
            r"(cleaning|damage|service) fee", r"charg(ed|ing).{0,15}\$?\d",
        ],
        default_risk="medium",
    ),
    Intent(
        key="CANCELLATION_FEE_DISPUTE",
        name="Cancellation Fee Dispute",
        description="Customer disputes a cancellation fee, or a driver cancelled / didn't show and the customer was still charged.",
        patterns=[
            r"cancel.{0,20}fee", r"fee.{0,20}cancel", r"cancel(led|ing)? (my|the) (ride|trip)",
            r"charged.{0,20}cancel", r"driver (cancel|never (showed|came)|no ?show)",
        ],
        default_risk="low",
    ),
    Intent(
        key="PROMO_REFERRAL_ISSUE",
        name="Promo / Referral / Coupon Issue",
        description="A promo code, referral credit, or coupon is not applying, expired unexpectedly, or a verification/OTP code isn't arriving.",
        patterns=[
            r"promo ?code", r"referral", r"coupon", r"discount code",
            r"verification code", r"(otp|one[- ]time (code|password))",
        ],
        default_risk="low",
    ),
    Intent(
        key="LOST_ITEM",
        name="Lost Item",
        description="Customer left a personal item in the vehicle and is trying to get it back.",
        patterns=[
            r"left (my|a|an) .{0,30}(in (the |his |her |their |your )?(car|cab|uber|vehicle)|behind)",
            r"lost (my|an?) (phone|wallet|bag|item|jacket|keys)",
            r"lost (and )?found", r"forgot my .{0,15}in",
        ],
        default_risk="medium",
    ),
    Intent(
        key="DRIVER_SAFETY_BEHAVIOR",
        name="Driver Safety / Behavior Incident",
        description="Customer reports unsafe driving, a safety incident, harassment, discrimination, rude/threatening behavior, or an accident.",
        patterns=[
            r"unsafe", r"rude", r"harass", r"threat(en(ed)?)?", r"assault",
            r"accident", r"crash", r"discriminat", r"racist", r"drunk driver",
            r"scared", r"felt unsafe", r"yell(ed|ing) at me",
            r"(photo|picture|profile).{0,20}(doesn'?t match|different)",
            r"(different|wrong) (car|driver).{0,15}(showed|arrived|than)",
        ],
        default_risk="high",
    ),
    Intent(
        key="TRIP_QUALITY_LOGISTICS",
        name="Trip Quality / Navigation / Wait",
        description="Driver got lost, took a bad route, long wait / ETA issues, wrong pickup or drop-off location, without a safety or billing complaint as the main point.",
        patterns=[
            r"got lost", r"wrong (way|route|turn|direction)", r"long wait",
            r"waiting (for|on) (my |the )?driver", r"wrong (pickup|pick[- ]up|drop ?off)",
            r"eta", r"driver (is |was )?late",
        ],
        default_risk="low",
    ),
    Intent(
        key="EATS_ORDER_ISSUE",
        name="Uber Eats Order Issue",
        description="Uber Eats specific: wrong/missing items, cold or late food, order cancelled, restaurant issue.",
        patterns=[
            r"uber ?eats", r"\beats\b", r"\border\b.{0,20}(wrong|missing|cold|late|cancel)",
            r"food (was |arrived )?(cold|late|missing)", r"missing item",
            r"restaurant",
        ],
        default_risk="low",
    ),
    Intent(
        key="APP_TECHNICAL_ISSUE",
        name="App / Technical Issue",
        description="The app itself is broken: crashing, won't load, payment method won't save, map glitches, feature not working -- not tied to a specific trip dispute.",
        patterns=[
            r"app (is |keeps |won'?t )?(crash|glitch|freez|not work|bug)",
            r"error message", r"won'?t (load|open|update)",
            r"payment method", r"can'?t add (a |my )?card",
        ],
        default_risk="low",
    ),
    Intent(
        key="DRIVER_PARTNER_SUPPORT",
        name="Driver / Partner Support",
        description="The person writing in is (or wants to become) a driver/courier, asking about driver-side pay, onboarding, or the driver app.",
        patterns=[
            r"become an? (uber )?driver", r"driver app", r"my earnings",
            r"as a driver", r"sign up to drive", r"driver pay",
        ],
        default_risk="medium",
    ),
    Intent(
        key="GENERAL_SUPPORT_UNRESPONSIVE",
        name="General Support Request / Unresponsive Support",
        description="Customer is asking for help or following up because a prior support request was ignored or is taking too long, without enough detail to place them in a more specific bucket.",
        patterns=[
            r"no(t| )?(one|body) (is )?(responding|helping|replying)",
            r"still (waiting|no response|no reply|haven'?t heard)",
            r"worst customer service", r"need (serious )?help",
            r"been (waiting|trying to reach)",
        ],
        default_risk="medium",
    ),
    Intent(
        key="PRAISE_OR_OFF_TOPIC",
        name="Praise / Off-topic / Non-actionable",
        description="Compliment, joke, meme, or a message with no actionable support request.",
        patterns=[
            r"\b(thank(s| you)|love|great job|awesome|amazing)\b",
            r"^\s*$",
        ],
        default_risk="low",
    ),
]

INTENTS_BY_KEY = {i.key: i for i in INTENTS}

_COMPILED = {
    i.key: [re.compile(p, re.IGNORECASE) for p in i.patterns] for i in INTENTS
}

# Evaluated roughly in this priority order: safety/security/billing/lost item
# are checked before the catch-all buckets, since a message can trivially
# match a generic word ("help") AND a specific one ("hacked") -- specific
# wins. Praise/off-topic is checked LAST as a true fallback-of-fallbacks
# only when nothing else, including "general support", matches.
PRIORITY = [
    "ACCOUNT_ACCESS_SECURITY", "DRIVER_SAFETY_BEHAVIOR", "LOST_ITEM",
    "BILLING_FARE_DISPUTE", "CANCELLATION_FEE_DISPUTE", "PROMO_REFERRAL_ISSUE",
    "EATS_ORDER_ISSUE", "APP_TECHNICAL_ISSUE", "DRIVER_PARTNER_SUPPORT",
    "TRIP_QUALITY_LOGISTICS", "GENERAL_SUPPORT_UNRESPONSIVE", "PRAISE_OR_OFF_TOPIC",
]


def rule_based_intent(text: str) -> str:
    """The 'trivial baseline' classifier: first matching keyword/regex rule
    wins, in PRIORITY order. Falls back to GENERAL_SUPPORT_UNRESPONSIVE."""
    for key in PRIORITY:
        for pat in _COMPILED[key]:
            if pat.search(text):
                return key
    return "GENERAL_SUPPORT_UNRESPONSIVE"
