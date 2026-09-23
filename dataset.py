import torch
import pandas as pd
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple, List, Union, Callable

class WSDDataset(torch.utils.data.Dataset):
    """
    Dataset for Word Sense Disambiguation (WSD) task. 
    This dataset generates features for token-based WSD tasks.
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, n_sense: int = 5,
            max_seq_length: int = 128) -> None:
        self.df = dataframe
        self.tokenizer = tokenizer
        self.n_sense = n_sense
        self.max_seq_length = max_seq_length

    def generate_features(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        sentence = row['sentence']
        correct_sense = row['correct_sense']

        base_senses = row.get('sense_list', [])[: max(0, self.n_sense - 1)]
        if correct_sense not in base_senses:
            base_senses.append(correct_sense)

        senses = base_senses[-self.n_sense:]
        assert len(senses) <= self.n_sense, f"Too many senses: {len(senses)}"
        senses = senses + [self.tokenizer.pad_token] * (self.n_sense - len(senses))

        encodded_sentence = self.tokenizer(
            sentence, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_seq_length
            )
        sentence_inputs = encodded_sentence["input_ids"].squeeze(0)
        sentence_mask = encodded_sentence["attention_mask"].squeeze(0)

        glosses_inputs, glosses_mask = [], []
        for sense in senses:
            encodded_sense = self.tokenizer(
                sense, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.max_seq_length
                )
            glosses_inputs.append(encodded_sense["input_ids"].squeeze(0))
            glosses_mask.append(encodded_sense["attention_mask"].squeeze(0))

        assert len(glosses_inputs) == len(glosses_mask) == self.n_sense
        labels = senses.index(correct_sense)
        return {
            "sentence_inputs": sentence_inputs,
            "sentence_mask": sentence_mask,
            "gloss_inputs": torch.stack(glosses_inputs),
            "gloss_mask": torch.stack(glosses_mask),
            "labels": torch.tensor(labels),
        }

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        row = self.df.iloc[index]
        return self.generate_features(row)

    def __len__(self) -> int:
        return len(self.df)
    


class SpanDataset(WSDDataset):
    """
    SpanDataset for Word Sense Disambiguation (WSD) task.
    This dataset generates features for span-based WSD tasks.
    """
    
    def generate_features(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """
        Generate QA like features with start and end positions of the correct sense in the context.
        """
        sentence = row['sentence']
        correct_sense = row['correct_sense']

        base_senses = row.get('sense_list', [])[: max(0, self.n_sense - 1)]
        if correct_sense not in base_senses:
            base_senses.append(correct_sense)

        senses = base_senses[-self.n_sense:]
        assert len(senses) <= self.n_sense, f"Too many senses: {len(senses)}"
        senses = senses + [self.tokenizer.pad_token] * (self.n_sense - len(senses))

        context = sentence + " " + " ".join(senses)
        start, end = self.find_start_and_end_position(correct_sense, context)
        assert start is not None and end is not None, "Start or end position not found"
        inputs = self.tokenizer(
            context, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_seq_length,
            add_special_tokens=False
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        start_positions = torch.tensor(start, dtype=torch.long)
        end_positions = torch.tensor(end, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }


    def sliding_window(self, sense: str, context: str) -> Tuple[int, int]:
        """
        Get the start and end positions of the query in the context.
        """
        query_ids = self.tokenizer.encode(sense, add_special_tokens=False)
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        start_pos, end_pos = None, None
        window_size = len(query_ids)
        for i in range(len(context_ids) - window_size + 1):
            if context_ids[i:i + window_size] == query_ids:
                start_pos = i
                end_pos = i + window_size - 1
                break
        return start_pos, end_pos


    def find_start_and_end_position(self, correct_sense: str, processed_sequence: str) -> Tuple[int, int]:
            start_char_index = processed_sequence.find(correct_sense)
            if start_char_index < 0:
                raise ValueError(f"Could not find `{correct_sense}` in `{processed_sequence}`.")
            end_char_index = start_char_index + len(correct_sense)

            encoding = self.tokenizer(
                processed_sequence,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            offset_mapping = encoding["offset_mapping"]

            start_token_index = None
            end_token_index = None
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_end == 0 and tok_start == 0:
                    continue
                # If the token’s span covers the first character of correct_sense,
                # that is our start_token.
                if tok_start <= start_char_index < tok_end:
                    start_token_index = i
                # If the token’s span covers the *last* character of correct_sense,
                # that is our end_token (inclusive).
                if tok_start <  end_char_index <= tok_end:
                    end_token_index = i
                    break

            return start_token_index, end_token_index
    



__all__ = ["WSDDataset", "SpanDataset"]