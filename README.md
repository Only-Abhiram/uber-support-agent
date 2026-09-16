# Uber AI Support Agent

This is my take-home submission for Hiver. I picked Uber, and I built a
small AI agent that looks at an incoming customer tweet and does three
things: figures out what it's about, drafts a reply, and decides whether
that reply is safe to send on its own or needs a human to check it first.

Below is a plain explanation of how it works, then how to run it.

## What it actually does, step by step

1. **Reads the message.** Just the customer's tweet, cleaned up a bit
   (I strip the `@Uber_Support` mention at the front, keep everything else
   — caps, emoji, punctuation, since those actually carry tone/urgency).

2. **Figures out the intent.** I looked at thousands of real complaints,
   clustered them to see what themes actually show up, and landed on 12
   categories — things like `BILLING_FARE_DISPUTE`, `LOST_ITEM`,
   `DRIVER_SAFETY_BEHAVIOR`, `PROMO_REFERRAL_ISSUE`. An LLM call (Gemini)
   picks the best-fitting one and also gives a confidence score.

3. **Finds similar past cases.** Before drafting anything, I search
   through real historical tweets of the same intent and pull the 2-3
   most similar ones, along with how Uber actually replied to them. No
   API call for this part — it's just local text similarity, so it's
   instant and free.

4. **Drafts a reply.** Gemini writes the actual reply, using those past
   examples as a style guide — not to copy facts from, just to match
   Uber's real tone and pattern. I'm explicit in the prompt: don't invent
   a refund amount or a promise you can't back up.

5. **Decides: send it, or flag a human?** A simple rulebook decides this
   — not another AI guess. Anything touching safety, threats, fraud, legal
   stuff, or a vulnerable customer always gets flagged, no matter how
   "confident" the AI was. Everything else gets auto-handled if the intent
   is low-risk and the model was confident enough.

## Why it's built this way (short version)

This dataset only has the *public* tweet exchange — the part where Uber
says "send us a DM" and then the real conversation moves somewhere I can't
see. So I didn't try to make the agent promise refunds or fix accounts —
it can't actually verify any of that. Its job here is to triage correctly
and sound like Uber while doing it, not to pretend it resolved your ticket.
(Longer version, with all the reasoning and the numbers, is in `REPORT.md`.)

## What's in each file, quickly

| File | What it's for |
|---|---|
| `src/data_prep.py` | Loads and cleans the raw conversation data |
| `src/intents.py` | The 12 categories + a keyword-based classifier (my "cheap" baseline) |
| `src/retrieval.py` | Finds similar past tweets to ground the reply in |
| `src/agent.py` | The real agent — talks to Gemini to classify + draft |
| `src/escalation_policy.py` | The send-it-or-flag-it rulebook |
| `src/llm_judge.py` | Grades the agent's own replies (separately, so it's an honest check) |
| `src/baselines.py` | Two simpler baselines I compare the agent against |
| `src/eval_harness.py` | Runs everything against my test set and scores it |
| `eval/golden_set.jsonl` | 204 examples I hand-labeled to check accuracy against |
| `REPORT.md` | The full write-up — results, what went wrong, what I'd do next |
| `DECISION_LOG.md` | The non-obvious calls I made, and why |

## Setup — the easy way

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your Gemini API key**
```bash
cp .env.example .env
```
Open `.env` and paste your key in. (Get one free at [Google AI Studio](https://aistudio.google.com/apikey) if you don't have one — and if you're reusing a key you've shared anywhere before, generate a fresh one.)

**3. Run the core pipeline** (no API key needed for this part — takes under 2 minutes)
```bash
./run_pipeline.sh
```
This loads the data, builds the retriever, and scores my two baselines. You'll see accuracy numbers print out at the end.

**4. Run the actual AI agent** (needs your API key + internet)
```bash
export GEMINI_API_KEY=your-key-here
python3 src/eval_harness.py --limit 10
```
This sends real messages through Gemini, gets real replies, and grades them. Takes about 5-10 minutes for 40 examples.

**5. (Optional but expected) Check the AI judge against your own judgment**
```bash
cd eval
python3 judge_agreement.py sample --n 30
```
This spits out a spreadsheet with 30 replies. Rate them yourself (1-5 on relevance, honesty, tone, helpfulness), then run:
```bash
python3 judge_agreement.py score
```
to see how much you and the AI judge agree.

That's it — everything else (the report, the decision log, the eval set) is already written and sitting in the repo, ready to read.

## One honest note

I built this in a sandboxed environment that couldn't actually reach
Gemini's servers, so I couldn't run step 4 myself and get real numbers
from the LLM agent — I only got to run and verify the non-AI parts
(baselines, retrieval, data prep). Everything in `REPORT.md` is marked
clearly as either "I actually ran this" or "you need to run this with your
key." Worth doing before I submit, so the numbers in the report are
actually mine and not placeholders.
