"""
Temporary FastAPI service to manually test the trained checkpoint
(output/step-12000-f1-0.9233.pt by default) against real questions.

Run:
    /home/synesis/venv/bin/uvicorn test_api:app --host 0.0.0.0 --port 8010

Then:
    POST /predict {"question": "..."}
        -> fetches candidate answers via the same top_k similarity API used
           to build the training data, then reranks them with the checkpoint.
    POST /qa {"question": "..."}
        -> same thing, but returns {"tag", "answer", ...} to match
           ec-bot-eval-scripts/evaluation_script_qa.py's expected contract.
"""
import os
import glob
import json
from typing import List

import torch
import requests
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from model import WSDModel
from utils import get_tokenizer

CONFIG_PATH = "config.yaml"
TAG_ANSWER_PATH = "data/tag_answer.json"
TOP_K_API = "http://20.51.201.105:8002/ec_bot/top_similar/"
TOP_K = 10

config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
MODEL_NAME = config["model_name"]
NUM_SENSE = config["num_sense"]
MAX_SEQ_LEN = config["max_seq_len"]
OUTPUT_DIR = config["output_dir"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(TAG_ANSWER_PATH, encoding="utf-8") as f:
    TAG_ANSWER = json.load(f)


def pick_best_checkpoint(output_dir: str) -> str:
    checkpoints = glob.glob(os.path.join(output_dir, "*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {output_dir}")
    return max(checkpoints, key=lambda p: float(p.rsplit("-", 1)[-1].removesuffix(".pt")))


tokenizer = get_tokenizer(MODEL_NAME)
model = WSDModel(MODEL_NAME, tokenizer=tokenizer, n_sense=NUM_SENSE).to(DEVICE)
checkpoint_path = pick_best_checkpoint(OUTPUT_DIR)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint: {checkpoint_path}")


def fetch_candidates(question: str, top_k: int = TOP_K) -> List[dict]:
    """Same retrieval step as prepare_data.py: similar questions -> their
    tags -> unique answers, in rank order. Keeps the tag/matched question so
    callers can report more than just the answer text."""
    resp = requests.post(TOP_K_API, json={"question": question, "top_k": top_k}, timeout=30)
    resp.raise_for_status()
    similar = resp.json()["top_similar"]
    candidates, seen_answers = [], set()
    for c in similar:
        answer = TAG_ANSWER.get(c["tag"])
        if answer and answer not in seen_answers:
            seen_answers.add(answer)
            candidates.append({
                "tag": c["tag"], "answer": answer,
                "matched_question": c["question"], "cosine_similarity": c["cosine_similarity"],
            })
    return candidates


@torch.no_grad()
def score_candidates(question: str, candidates: List[str]) -> List[float]:
    """Scores an arbitrary number of candidates by running them through the
    model in windows of NUM_SENSE (the fixed gloss-slot count the checkpoint
    was trained with), padding the last window if needed, then normalizing
    all collected logits together with a single softmax."""
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

    probs = torch.softmax(torch.tensor(all_logits), dim=0).tolist()
    return probs


app = FastAPI(title="FAQ WSD checkpoint tester (temporary)")


class PredictRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class CandidateScore(BaseModel):
    answer: str
    score: float


class PredictResponse(BaseModel):
    question: str
    predicted_answer: str
    ranked_candidates: List[CandidateScore]


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": checkpoint_path, "device": DEVICE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    candidates = fetch_candidates(req.question, req.top_k)

    if not candidates:
        return PredictResponse(question=req.question, predicted_answer="", ranked_candidates=[])

    answers = [c["answer"] for c in candidates]
    scores = score_candidates(req.question, answers)
    ranked = sorted(zip(answers, scores), key=lambda x: x[1], reverse=True)
    return PredictResponse(
        question=req.question,
        predicted_answer=ranked[0][0],
        ranked_candidates=[CandidateScore(answer=a, score=s) for a, s in ranked],
    )


class QARequest(BaseModel):
    question: str
    top_k: int = TOP_K


CONFIDENCE_THRESHOLD = 0.5


@app.post("/qa")
def qa(req: QARequest):
    """Matches evaluation_script_qa.py's expected `/qa` contract: takes a bare
    question, returns at least `tag` and `answer`."""
    candidates = fetch_candidates(req.question, req.top_k)
    if not candidates:
        return {
            "tag": "unable_to_answer", "answer": "", "confident": False,
            "top_cosine": None, "matched_question": "",
        }

    answers = [c["answer"] for c in candidates]
    scores = score_candidates(req.question, answers)
    best_index = max(range(len(answers)), key=lambda i: scores[i])
    best = candidates[best_index]
    return {
        "tag": best["tag"],
        "answer": best["answer"],
        "confident": scores[best_index] >= CONFIDENCE_THRESHOLD,
        "top_cosine": candidates[0]["cosine_similarity"],
        "matched_question": best["matched_question"],
    }


@app.post("/rank")
def rank(req: QARequest):
    """Debug endpoint: full ranked candidate pool with tags + scores, to
    diagnose whether a miss is a retrieval failure (correct tag never
    surfaced by the top_k API) or a reranking failure (it was there, scored
    lower than a distractor)."""
    candidates = fetch_candidates(req.question, req.top_k)
    if not candidates:
        return {"question": req.question, "candidates": []}
    answers = [c["answer"] for c in candidates]
    scores = score_candidates(req.question, answers)
    ranked = sorted(
        [{"tag": c["tag"], "answer": c["answer"], "score": s, "cosine_similarity": c["cosine_similarity"]}
         for c, s in zip(candidates, scores)],
        key=lambda x: x["score"], reverse=True,
    )
    return {"question": req.question, "candidates": ranked}
