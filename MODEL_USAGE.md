# Using the FAQ reranker checkpoint in ec-vectordb-service

This describes the model trained in this repo (`csebuetnlp/banglabert`, cosine
architecture, `output/step-12000-f1-0.9233.pt`) and how to call it from
another service. A working reference implementation already exists in this
repo at `test_api.py` (`/qa` and `/rank` endpoints) — read that alongside this
doc if anything here is ambiguous.

## What this model is

A **reranker**, not a classifier and not a generator. Given one question and
a small set of *candidate answer texts*, it scores each candidate and returns
the one it thinks answers the question. It never produces a tag or an answer
on its own — it only picks among candidates you hand it.

It cannot replace your retrieval step. You still need something upstream
(FAISS / embedding similarity / the existing `top_k` retrieval) to produce
the candidate pool per question. This model's whole job is: given that pool,
pick the right one more reliably than raw cosine similarity does alone.

## What it isn't good at yet (read before wiring it in)

Measured on the 7,406-turn `gold_eval.xlsx` gold set (see `eval_output/` and
the earlier eval run in this repo):

- **87.3% correct-tag rate overall** (88.5% on one-turn scenarios, 70.2% on
  multi-turn scenarios scored without conversation history — this model is
  **stateless**, it has no concept of prior turns).
- **0 errors, ~0.3% false abstentions** — it essentially always answers
  *something*, it just isn't always right.
- Of the failures, **93.3% are cases where the correct answer was in the
  candidate pool but scored lower than a distractor** — a genuine reranking
  mistake, not a retrieval gap. 77.6% of those are literally the runner-up
  (rank 2), almost always in a pool of only 2-3 candidates, i.e. **head-to-head
  confusion between two near-duplicate sibling FAQ tags**
  (e.g. `otp_entry_screen` vs `otp_send_button`,
  `nid_wallet_download_iphone` vs `nid_wallet_download`,
  `liveness_check_avoid` vs `liveness_check_steps`).
- These wrong picks are usually **confident, not uncertain**: median score
  gap between the (wrong) top pick and the (correct) runner-up is 0.95.
  Do not treat this model's score as a reliable "how sure am I" signal for
  abstention thresholds without recalibrating it on your own data.

Root cause: 61% of training rows had only one candidate answer during
training (no contrastive/hard-negative practice), so the model never learned
to separate those specific sibling pairs. If you hit accuracy problems in
production, augmenting training data with deliberate hard negatives for
confusable tag pairs (rather than retraining on more of the same) is the
likely fix — ask before assuming a bigger model or more epochs is the answer.

## Files you need

From `output/`: the checkpoint, e.g. `step-12000-f1-0.9233.pt` (~440MB).
Pick the highest-F1 `.pt` file if more than one exists — filenames encode F1
as `step-<n>-f1-<score>.pt`.

The base model `csebuetnlp/banglabert` downloads from the HuggingFace Hub on
first load (~110M params, ELECTRA, 32k vocab) — no separate file to ship for
that part, just network access to `huggingface.co` (or a mirror/cache) at
startup.

## Config values baked into this checkpoint

These are not optional at inference time — they must match what the
checkpoint was trained with, from `config.yaml`:

| key | value |
|---|---|
| `model_name` | `csebuetnlp/banglabert` |
| `num_sense` | `3` |
| `max_seq_len` | `96` |

`num_sense=3` is the important one: the model's architecture hard-codes a
fixed 3-slot comparison per forward pass (see "Scoring more than 3
candidates" below for how to work around this for larger pools).

## Loading the model

```python
import torch
from transformers import AutoTokenizer
from model import WSDModel  # from this repo

MODEL_NAME = "csebuetnlp/banglabert"
NUM_SENSE = 3
MAX_SEQ_LEN = 96
CHECKPOINT_PATH = "output/step-12000-f1-0.9233.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"additional_special_tokens": ["<classify>", "</classify>"]})

model = WSDModel(MODEL_NAME, tokenizer=tokenizer, n_sense=NUM_SENSE).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
```

The `add_special_tokens` call **must** happen before building `WSDModel` —
`WSDModel.__init__` calls `resize_token_embeddings(len(tokenizer))`, and the
checkpoint's embedding matrix was saved at that resized size. Skipping it (or
changing the token count) will fail to load the state dict.

## Scoring candidates

Input: one question (str) + a list of candidate answer texts (str). Output:
a probability per candidate, comparable across the whole list even though
the model only ever compares 3 at a time internally.

```python
import torch

@torch.no_grad()
def score_candidates(question: str, candidates: list[str]) -> list[float]:
    encoded_sentence = tokenizer(
        question, return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_SEQ_LEN,
    )
    sentence_inputs = encoded_sentence["input_ids"].to(DEVICE)
    sentence_mask = encoded_sentence["attention_mask"].to(DEVICE)

    all_logits = []
    for start in range(0, len(candidates), NUM_SENSE):
        window = candidates[start:start + NUM_SENSE]
        padded = window + [tokenizer.pad_token] * (NUM_SENSE - len(window))

        encoded = tokenizer(
            padded, return_tensors="pt", padding="max_length",
            truncation=True, max_length=MAX_SEQ_LEN,
        )
        gloss_inputs = encoded["input_ids"].unsqueeze(0).to(DEVICE)
        gloss_mask = encoded["attention_mask"].unsqueeze(0).to(DEVICE)

        outputs = model(
            sentence_inputs=sentence_inputs, sentence_mask=sentence_mask,
            gloss_inputs=gloss_inputs, gloss_mask=gloss_mask,
        )
        all_logits.extend(outputs.logits[0][: len(window)].tolist())

    return torch.softmax(torch.tensor(all_logits), dim=0).tolist()
```

Pick the answer: `candidates[max(range(len(candidates)), key=lambda i: scores[i])]`.

### Scoring more than 3 candidates

The loop above windows the candidate list into groups of `NUM_SENSE` (3),
padding the last group if needed, and normalizes all collected logits
together with one final softmax. This is a workaround for the fixed 3-slot
architecture, not the original training setup — the model was only ever
directly compared against exactly 3 at training time (1 correct + up to 2
distractors). Treat rankings among candidates from *different* windows as
less trustworthy than rankings within the same window; if you can keep your
candidate pool at ≤3 per question, do that instead of relying on this
windowing trick.

## Suggested integration point

Use this as a **reranking pass on top of your existing retrieval**, not a
replacement for it:

1. Run your existing retrieval (FAISS / embedding voting / whatever
   `ec-vectordb-service`'s `/qa` already does) to get a small set of
   candidate `(tag, answer)` pairs — same shape as the `top_k` API this model
   was trained against (`http://20.51.201.105:8002/ec_bot/top_similar/`,
   returns `{tag, question, cosine_similarity}` per candidate).
2. Feed the question + candidate answer texts to `score_candidates`.
3. Take the top-scoring candidate's tag/answer as your response.

Keep your existing retrieval's own top-1 pick around as a fallback/tiebreaker
signal — this model is not calibrated for abstention decisions on its own
(see limitations above), so don't wire its score directly into a
"decline to answer" threshold without validating that threshold on your own
gold set first.

## Reference implementation

`test_api.py` in this repo is a working FastAPI service implementing this
exact contract:

- `POST /qa {"question": "..."}` → `{"tag", "answer", "confident", ...}`,
  matching `ec-bot-eval-scripts/evaluation_script_qa.py`'s expected `/qa`
  contract, useful as a direct integration reference.
- `POST /rank {"question": "..."}` → full ranked candidate list with tags
  and scores, useful for debugging specific misroutes.

Run it locally with `uvicorn test_api:app --port 8010` to see live examples
before porting the logic into `ec-vectordb-service`.
