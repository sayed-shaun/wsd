import asyncio
import csv
import json
import sys

import aiohttp
from tqdm.asyncio import tqdm

CSV_PATH = "/home/synesis/Desktop/ec-faq-bot/full_dataset/question_tag.csv"
TOP_K_API = "http://20.51.201.105:8002/ec_bot/top_similar/"
OUTPUT_PATH = "/home/synesis/Desktop/ec-span-classifier/prepared_data.json"
TOP_K = 10
MAX_RETRIES = 3
CONCURRENCY = 20


def load_questions(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [(row["question"], row["tag"]) for row in reader]


async def query_top_similar(session, question, top_k=TOP_K):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(
                TOP_K_API,
                json={"question": question, "top_k": top_k},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["top_similar"]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == MAX_RETRIES:
                print(f"  API call failed after {MAX_RETRIES} attempts: {e}", file=sys.stderr)
                return None
            await asyncio.sleep(2 * attempt)


def dedup_candidates(candidates, top_k=TOP_K):
    """Keep only distinct candidates by question text, capped at top_k."""
    seen = set()
    distinct = []
    for c in candidates:
        if c["question"] not in seen:
            seen.add(c["question"])
            distinct.append(c)
        if len(distinct) == top_k:
            break
    return distinct


def pick_right_candidate(candidates, true_tag):
    """Among the top-k candidates, pick the one matching the question's known
    tag with the highest cosine similarity. Falls back to rank 1 (flagged
    with index -1) if no candidate shares the ground-truth tag."""
    matches = [c for c in candidates if c["tag"] == true_tag]
    if matches:
        best = max(matches, key=lambda c: c["cosine_similarity"])
        index = candidates.index(best)
        return best, index
    return candidates[0], -1


async def process_row(session, semaphore, question, true_tag):
    async with semaphore:
        candidates = await query_top_similar(session, question)
    if not candidates:
        return None

    candidates = dedup_candidates(candidates)
    right_candidate, index = pick_right_candidate(candidates, true_tag)
    return {
        "question": question,
        "true_tag": true_tag,
        "right_candidate": right_candidate,
        "top_k_candidates": candidates,
        "index": index,
    }


async def main_async():
    rows = load_questions(CSV_PATH)
    total = len(rows)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results_by_index = [None] * total

    async with aiohttp.ClientSession() as session:
        async def worker(i, question, true_tag):
            results_by_index[i] = await process_row(session, semaphore, question, true_tag)

        tasks = [
            asyncio.create_task(worker(i, question, true_tag))
            for i, (question, true_tag) in enumerate(rows)
        ]

        completed = 0
        for coro in tqdm.as_completed(tasks, total=total, desc="Querying top_similar"):
            await coro
            completed += 1
            if completed % 200 == 0:
                results = [r for r in results_by_index if r is not None]
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    results = [r for r in results_by_index if r is not None]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    unmatched = sum(1 for r in results if r["index"] == -1)
    print(f"\nDone. Wrote {len(results)}/{total} records to {OUTPUT_PATH}")
    print(f"{unmatched} questions had no top-k candidate matching their ground-truth tag (flagged index=-1)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
