# Decision Log

Non-obvious calls made while building this, and why. (Bullet points, per
the assignment brief.)

1. **Anchored on the first customer message + first brand reply only, not
   the full thread.** Later turns in this dataset are almost always the
   customer moving to DM per the brand's own request, so turn 3+ has no
   text we can see. Modeling deeper turns would mean training/evaluating
   on a biased subset (threads where the customer stayed public), which is
   worse than being honest about the boundary.

2. **Used unsupervised clustering to discover the taxonomy, then hand-curated
   the final 12 intents rather than shipping the raw cluster IDs.**
   TF-IDF + LSA + KMeans on tweet-length text is noisy (silhouette scores
   topped out around 0.03-0.04 across k=8..24 -- see `src/intents.py`
   docstring). One 14-cluster run gave a single catch-all cluster holding
   ~32% of the sample. Rather than force a clean unsupervised answer that
   isn't there, clustering was used for discovery (what are the recurring
   themes?) and a human made the final taxonomy + labeling-rule call. Being
   upfront that this is a hybrid, not a pure unsupervised pipeline.

3. **The "trivial baseline" (regex/keyword rules) is also what bootstraps
   the "simple baseline" (TF-IDF + LogisticRegression) training labels.**
   Hand-labeling 40k+ examples wasn't feasible in the time available. This
   is weak supervision, and it shows: on the held-out, hand-corrected
   golden set, the trained classifier (68.6% accuracy) actually
   *underperforms* the rule tagger it was trained to imitate (77.5%
   accuracy) -- see REPORT.md. That's a real, useful finding, not a bug:
   it says the classifier mostly learned to reproduce the rule tagger's
   blind spots plus added noise, and that a bigger win would come from
   fixing/expanding the weak-label source or hand-labeling more data,
   not from a fancier classifier on the same labels.

4. **Retrieval-for-grounding uses local TF-IDF cosine similarity, not an
   embedding API call.** Keeps the expensive/rate-limited part of the
   system (LLM calls) to exactly 2 per message (classify, draft), keeps
   retrieval fully offline and reproducible, and is good enough at
   finding near-duplicate complaints (see the retrieval.py demo: 0.4+
   cosine similarity on "charged twice" queries).

5. **"Auto-handle" was redefined, deliberately, to mean "send the correct
   on-brand triage/acknowledgement," not "resolve the ticket."** This
   dataset's brand replies are almost entirely templated deflections to
   DM ("send us a note..." showed up as the top of the reply-text
   frequency table for >70% of a 5k sample). The agent has no access to
   account/trip/payment systems here, so promising a refund or account
   restore would be fabrication, not automation. Reframing "good" this
   way is argued in REPORT.md's "Problem framing" section.

6. **The golden set's intent corrections and escalation ground truth were
   AI-assisted (an LLM read all 204 candidate rows and labeled them), not
   independently multi-rater human-labeled.** This is flagged explicitly
   rather than glossed over, because the assignment will be defended live
   and because an unflagged AI-labeled "golden" set is exactly the kind of
   quietly-wrong headline number the brief's "what's misleading" section
   is designed to catch. Recommendation to whoever submits this: spot-check
   at least 30-40 of the 204 rows yourself before relying on the numbers.

7. **Classifier "confidence" for the LLM path is the model's own
   self-reported 0-1 estimate, not a calibrated softmax probability.**
   There's no local logits access via the Gemini API in the way this
   pipeline calls it. This is treated as a rough signal only -- the
   escalation policy uses it as one input among several (hard triggers +
   intent risk tier), not the sole gate, specifically because self-reported
   LLM confidence is known to be poorly calibrated.

8. **Escalation policy checks hard safety/legal/vulnerability regex
   triggers BEFORE consulting the predicted intent or its confidence.**
   A message that got classified as BILLING_FARE_DISPUTE but mentions
   "lawyer" or "assault" should still escalate regardless of how confident
   the classifier is in the (wrong-ish) label. Classification errors
   should not be able to suppress a safety escalation.

9. **Stratified, not random, sampling for the golden set** -- capped/boosted
   per rule-predicted bucket so rare-but-important intents (lost item,
   driver safety, account security) get enough eval examples to be
   measurable, instead of a random sample where a 69%-of-corpus catch-all
   bucket would dominate and safety-critical classes would get ~5 examples
   each. Directly causes the "misleading headline number" callout in
   REPORT.md: metrics computed on this set are NOT population-representative.

10. **Only 2 LLM calls per message (classify, draft) plus 1 separate judge
    call at eval time**, rather than one combined call doing everything.
    Keeping classification and drafting separate means retrieval can sit
    between them (the draft prompt needs the predicted intent to know
    which examples to retrieve), and keeping the judge as an independent
    call means it can't see the retrieval context the drafter saw, so it's
    actually grading the output rather than checking self-consistency.

11. **The reply-quality rubric's "faithfulness" dimension is graded
    separately from "relevance," and the judge prompt explicitly says a
    vague-but-honest reply should score HIGH on faithfulness.** Without
    this instruction, LLM judges reliably reward confident, specific-
    sounding replies even when the specifics are made up -- exactly the
    failure mode that matters most for a support agent that can't actually
    see the customer's trip/account data.

12. **A rule-based "hard trigger" list (violence, self-harm, legal threats,
    minors, ADA/accessibility, media/PR, discrimination) always forces
    escalation, overriding everything else in the pipeline**, including a
    high-confidence low-risk intent classification. This is a deliberate,
    conservative floor: it will produce some unnecessary escalations
    (e.g. a message that mentions "lawyer" sarcastically), which is an
    acceptable cost given the alternative failure mode.

13. **The repo does not commit the full raw Kaggle dataset** (redistribution
    of a Kaggle-licensed dataset in a public repo is avoided), only a
    2,000-row sample under `data/sample/` for a no-download quick start,
    plus a documented schema so anyone can point the pipeline at their own
    downloaded copy.

14. **No embedding model or local LLM was downloaded for clustering** --
    the sandbox this was built in only has network egress to a small
    allowlist (pypi, npm, github, etc.), not to Hugging Face Hub or
    Google's Gemini endpoint. TF-IDF/LSA/KMeans was chosen partly because
    it's genuinely a reasonable first pass for short noisy text, and
    partly because it's what could actually be run and verified end-to-end
    in the build environment. See REPORT.md and README.md for exactly
    which numbers in this report were actually executed vs. designed-and-
    documented for you to run with your own Gemini API key.
