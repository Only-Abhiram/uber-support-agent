# Report: Uber AI Support Agent

## 1. Problem framing

**What data we actually have.** The dataset is the *public* half of Uber's
Twitter support threads: a customer's tweet and Uber's public reply. A
quick frequency check over the first 5,000 threads shows the brand's
first-touch reply is dominated by a handful of near-identical templates --
"Here to help. Send us a note at [link] so our team can connect," "Send us
a DM with your email address," etc. -- accounting for the majority of
replies. The *actual* resolution (a refund issued, an account restored, a
fare adjusted) happens afterward in a private DM this dataset does not
contain.

**What "good" means here, given that.** We deliberately did NOT build a
system that pretends to resolve refunds, restore accounts, or issue
compensation -- doing so on this data would mean the agent is fabricating
outcomes it has no way to verify (no access to trip, payment, or account
systems). Instead, "good" is defined at the layer this data actually
supports -- **the triage layer**:
- Correctly identify *what kind* of problem this is.
- Draft a reply that is warm, on-brand, specific to what the customer
  actually said, and asks for exactly the right next piece of information
  -- without inventing a refund amount, timeline, or cause.
- Correctly flag the subset of messages a human must see before anything
  goes out (safety, legal, fraud, vulnerable customers, or cases the model
  itself isn't confident about) rather than let an AI auto-reply to a
  domestic-violence-adjacent tweet with a cheerful template.

**What we chose not to build:**
- A system that promises specific refunds/compensation (fabrication risk).
- Multi-turn dialogue management (we don't have the DM turns to learn from).
- A single "one LLM call does everything" agent -- classification,
  retrieval, and drafting are kept as separate, individually-testable
  stages (see DECISION_LOG.md #10).
- A perfectly clean unsupervised intent taxonomy -- the data doesn't
  support one at k this small (see Results below), so the taxonomy is
  hybrid: clustering for discovery, human judgment for the final label set.

## 2. The taxonomy (discovered, then curated)

TF-IDF (1-2 grams) on the first customer message -> TruncatedSVD (100
comps, ~17% variance explained) -> KMeans, k swept 8-24, silhouette-selected
(all scores were low, 0.02-0.04 -- see Section 5). k=14 was inspected by
hand (top terms + 4 random raw examples per cluster); one cluster held
~32% of the sample with no coherent top terms (a true "everything else"
bucket), and several others were the same underlying complaint split by
surface wording. These were merged/split by hand into 12 final intents:

`ACCOUNT_ACCESS_SECURITY, BILLING_FARE_DISPUTE, CANCELLATION_FEE_DISPUTE,
PROMO_REFERRAL_ISSUE, LOST_ITEM, DRIVER_SAFETY_BEHAVIOR,
TRIP_QUALITY_LOGISTICS, EATS_ORDER_ISSUE, APP_TECHNICAL_ISSUE,
DRIVER_PARTNER_SUPPORT, GENERAL_SUPPORT_UNRESPONSIVE, PRAISE_OR_OFF_TOPIC`

Full descriptions + the regex rules used to bootstrap labels are in
`src/intents.py`. Rule-tagged distribution over all 42,680 threads:

| Intent | % of corpus |
|---|---|
| GENERAL_SUPPORT_UNRESPONSIVE (catch-all) | 69.0% |
| BILLING_FARE_DISPUTE | 8.2% |
| EATS_ORDER_ISSUE | 5.5% |
| CANCELLATION_FEE_DISPUTE | 3.6% |
| PRAISE_OR_OFF_TOPIC | 3.5% |
| DRIVER_SAFETY_BEHAVIOR | 3.0% |
| ACCOUNT_ACCESS_SECURITY | 2.5% |
| PROMO_REFERRAL_ISSUE | 1.8% |
| TRIP_QUALITY_LOGISTICS | 1.3% |
| LOST_ITEM / APP_TECHNICAL_ISSUE | 0.7% each |
| DRIVER_PARTNER_SUPPORT | 0.3% |

That 69% catch-all is itself a finding, not just a gap: short, noisy,
one-off tweets resist keyword rules, which is the core argument for the
LLM classifier over the trivial baseline (Section 4).

## 3. Golden evaluation set

204 examples, stratified-sampled across the 12 rule-predicted buckets
(over-sampling small classes so rare-but-important intents like
`LOST_ITEM` and `DRIVER_SAFETY_BEHAVIOR` have enough eval mass to be
measurable -- a uniform random sample would have given ~5 examples to
each safety-relevant class). Every example was then read and the intent
label was confirmed or corrected by hand (46/204 = 22.5% were corrected),
and each got a hand-assigned escalation ground truth (112 ESCALATE / 92
AUTO_HANDLE) judged directly against the policy categories in
`src/escalation_policy.py`. Full methodology and the actual correction
list (useful as a list of real rule-tagger failure cases) are in
`eval/build_golden_set.py`. **This labeling pass was AI-assisted, not
independently multi-rater human-labeled -- see Decision Log #6 and Section
6 below for why that matters for trusting these numbers.**

## 4. Results vs. baselines

Two baselines, both actually run and scored against the golden set
(`src/baselines.py`, output in `eval/results/`):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Trivial** (regex/keyword rules) | 0.775 | 0.756 | 0.801 |
| **Simple** (TF-IDF + LogisticRegression, trained on rule pseudo-labels) | 0.686 | 0.685 | 0.719 |
| **Agent** (Gemini, few-shot over the taxonomy) | *run `eval_harness.py` with your API key* | | |

**The simple baseline underperforms the trivial one it was trained to
imitate.** This is a genuine, if slightly awkward, result: training a
classifier on weakly-supervised labels from the rule tagger doesn't beat
the rule tagger, measured against hand-corrected ground truth -- it mostly
learns to reproduce the rule tagger's own blind spots (see
`DRIVER_SAFETY_BEHAVIOR` recall drop 0.68 -> 0.56 in the per-class reports
below) while adding its own generalization noise elsewhere. The fix isn't
a better classifier on the same labels; it's better labels (more hand
labeling, or a better weak-label source -- which is exactly the gap the
LLM classifier is meant to fill).

Per-class detail (support = count in the 204-example golden set):

**Trivial baseline**, worst classes: `PRAISE_OR_OFF_TOPIC` (precision 0.13
-- catches lots of false positives from generic thank-you/help phrasing),
`GENERAL_SUPPORT_UNRESPONSIVE` (precision 0.40 -- the catch-all catches
too much), `TRIP_QUALITY_LOGISTICS` (precision 0.40, recall 0.50).
Full report: `eval/results/trivial_report.txt`.

**Simple baseline**, same weak spots plus a meaningfully worse
`DRIVER_SAFETY_BEHAVIOR` recall (0.56 vs 0.68) -- worrying, since this is
the class where a missed classification is most likely to also mean a
missed escalation. Full report: `eval/results/simple_report.txt`.

**Agent (Gemini)**: fully implemented (`src/agent.py`, `src/eval_harness.py`)
but not executable in the sandbox this was built in (no network route to
`generativelanguage.googleapis.com` -- confirmed via a direct request that
returned `x-deny-reason: host_not_allowed`). Run `eval_harness.py`
yourself (~5-10 min, see README) to fill in this row, plus the LLM-judge
reply-quality scores and their human-agreement check.

## 5. Failure analysis: top 5 failure modes

1. **Keyword false positives from figurative/idiomatic language.** The
   rule tagger's `crash`/`crashed` pattern (meant for vehicle accidents)
   fired on "HAS CRASHED... WE NEED YOUR SWEET SWEET DOUGHNUTS" (a meme)
   and "uber eats crashes at 2pm" (the *app* crashing). Both were
   mislabeled `DRIVER_SAFETY_BEHAVIOR` by the trivial baseline. Hypothesis:
   any keyword-only system will keep hitting this; an LLM classifier
   should not, since it has the surrounding context.

2. **Safety-relevant content hiding behind non-safety phrasing.** Several
   golden-set corrections were messages a keyword scanner would never flag
   as safety-related on vocabulary alone: "he kept asking me questions
   about my headdress, insulted me & touched my knee" (harassment, no
   safety keyword), "I think my account is hacked" phrased as "email being
   used by someone in Russia" (account takeover, no "hack" keyword), "so
   shaken up from this order" (a courier redirect attempt, no threat
   keyword). This is the most consequential failure mode: it's a false
   *negative* on escalation-worthy content, not just a labeling
   inconvenience.

3. **The catch-all intent absorbs everything a rule doesn't recognize,
   regardless of actual severity.** 69% of the full corpus falls into
   `GENERAL_SUPPORT_UNRESPONSIVE` under the trivial tagger. Some of that
   is genuinely generic ("hello"), but plenty is a specific complaint
   phrased without the exact keywords the rules look for (e.g. "left my
   phone in one of your drivers cars" was missed because the lost-item
   pattern expected "in the car" not "in ... cars").

4. **Weak-label training compounds rather than corrects rule-tagger
   blind spots.** The simple baseline's `DRIVER_SAFETY_BEHAVIOR` recall
   dropped further than the rule tagger's own, because it was trained to
   imitate the rule tagger on exactly this class. Any weakly-supervised
   pipeline inherits and can amplify its label source's specific failure
   modes -- worth remembering before trusting a "we trained a classifier
   so it must be better" headline number.

5. **Multi-issue messages force a single-label choice that loses
   information.** E.g. "3 drivers cancelled, 2nd one played an insulting
   song, charged me $5" mixes a cancellation fee, inappropriate driver
   content, and a billing dispute in one tweet. A single intent label
   (we chose `DRIVER_SAFETY_BEHAVIOR`, the most severe) is defensible for
   escalation routing but throws away the billing detail a human agent
   would still need. Hypothesis: a production system should support
   multi-label output, not force single-label classification.

## 6. What is misleading about my headline number(s)

This section is mandatory and deserves to be read as carefully as the
results table:

- **The golden set is stratified, not representative.** We deliberately
  over-sampled rare intents (lost item, driver safety, account security)
  so they'd be measurable at all. That means: (a) the 77.5%/68.6% accuracy
  numbers above are NOT the accuracy you'd get running either baseline on
  a random sample of real traffic (the true traffic is 69% one catch-all
  bucket the trivial tagger is only ~50% precise on -- see Section 5 #3 --
  so population-level accuracy is likely noticeably lower than 77.5%), and
  (b) the 112-ESCALATE / 92-AUTO_HANDLE split in the golden set massively
  overstates how often real traffic needs escalation -- `DRIVER_SAFETY_
  BEHAVIOR` and `ACCOUNT_ACCESS_SECURITY` are only ~5.5% of the real corpus
  combined, not the ~35%+ they occupy in this stratified sample.

- **The golden set's "hand" labels were AI-assisted, not independently
  human-verified.** One careful reader (an LLM, in this case) went through
  all 204 rows once. There was no second rater, no adjudication, no
  inter-rater reliability check. Treat the 22.5% "trivial baseline error
  rate" as a reasonable estimate, not a certified number, until someone
  spot-checks a sample of it by hand.

- **"Grounded in how the brand has historically resolved similar issues"
  is weaker than it sounds.** As explained in Section 1, the retrieved
  "historical resolutions" are almost always deflection templates, not
  actual resolutions. The agent is grounded in the brand's historical
  *triage voice*, not in evidence about what outcomes those customers
  actually got. Don't read "grounded" as "verified against real resolution
  outcomes" -- it isn't, because this dataset doesn't contain outcomes.

- **An LLM-judge score, once you get one, is not ground truth either.**
  That's exactly why the human-agreement check (`eval/judge_agreement.py`)
  is a required part of this deliverable, not an optional nice-to-have --
  and why it hasn't been faked with fabricated agreement numbers here. If
  you skip actually filling in the human ratings, don't quote the judge's
  mean scores as if they were validated.

- **Classifier confidence, for the LLM path, is self-reported by the same
  model doing the classifying**, not a calibrated probability (Decision
  Log #7). A confidently-wrong classification is possible and the
  escalation policy only partially guards against it (via hard-trigger
  regexes that don't depend on the classifier at all).

## 7. What we'd do next with one more week

1. **Independent human labeling of the golden set** (2-3 raters, measure
   inter-rater agreement, adjudicate disagreements) to replace the
   AI-assisted single-pass labels with something defensible as ground truth.
2. **Fix the weak-label bottleneck directly**: either hand-label a larger
   seed set (1-2k examples) to train the simple baseline on, or use the
   LLM classifier itself (once validated) to re-label the full corpus and
   retrain a cheap classifier on *those* labels for a low-cost production
   path that doesn't call an LLM on every message.
3. **Multi-label classification** for genuinely multi-issue messages
   (Section 5 #5), so escalation routing and reply drafting can both see
   the full issue set, not a forced single label.
4. **A held-out temporal split** (train/eval on different time windows)
   to check the taxonomy and classifiers aren't overfit to a particular
   period's slang/promo-code campaigns/etc.
5. **Real calibration of the escalation threshold** using the
   precision/recall tradeoff on the escalate class specifically (not
   overall accuracy), once the agent's numbers exist -- right now
   `CONFIDENCE_THRESHOLD = 0.55` in `escalation_policy.py` is a reasonable
   starting guess, not a tuned value.
6. **A second LLM-judge rubric pass specifically for the safety-relevant
   intents**, since Section 5's failure modes suggest generic reply
   quality and safety-appropriate handling may need separate scoring --
   a reply can be relevant/faithful/well-toned by the generic rubric while
   still being wrong for a harassment report (e.g. too casual).

---
See `DECISION_LOG.md` for the full list of build decisions and
`README.md` for exact reproduction steps.
