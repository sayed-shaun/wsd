import json

INPUT_FILE = "data/prepared_data.json"
ANSWER_FILE = "data/tag_answer.json"
OUTPUT_FILE = "data/wsd_train_data.json"
TOP_K = 10  # unique candidate answers kept from the top_k API results

with open(INPUT_FILE) as f:
    data = json.load(f)

with open(ANSWER_FILE) as f:
    tag_answer = json.load(f)

records = []
for entry in data:
    sentence = entry["question"]
    true_tag = entry["true_tag"]
    correct_sense = tag_answer[true_tag]

    # sense_list holds the unique top_k answers as-is; correct_sense is not
    # allowed to steal one of those slots. dataset.py guarantees correct_sense
    # ends up in the trained-on sense pool even if it's absent here.
    sense_list = []
    for c in entry["top_k_candidates"]:
        answer = tag_answer[c["tag"]]
        if answer not in sense_list:
            sense_list.append(answer)
        if len(sense_list) == TOP_K:
            break

    records.append(
        {
            "sentence": sentence,
            "sense_list": sense_list,
            "correct_sense": correct_sense,
        }
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(records)} records to {OUTPUT_FILE}")
