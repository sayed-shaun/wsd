import os
import json
from typing import Union
import nltk
import random
import pandas as pd
import polars as pl
import xml.etree.ElementTree as ET
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn
from wn_utils import fill_empty_sense, get_correct_sense, get_all_senses


def load_wsd_json(path: str) -> pd.DataFrame:
    """
    Load a pre-built WSD-style gold file (list of {sentence, sense_list,
    correct_sense} records, e.g. data/wsd_train_data.json) into a dataframe.
    Built directly as pandas (not via polars) so `sense_list` stays a plain
    Python list per row instead of becoming a numpy.ndarray, which breaks the
    list `.append()`/`+` operations in dataset.py's WSDDataset/SpanDataset.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def generate(dir_path: str) -> pl.DataFrame:
    """
    This function extract data from xml file and return a pandas dataframe.
    https://github.com/HSLCY/GlossBERT/blob/master/preparation/generate.py
    """
    file_name = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if file.endswith(".xml")][0]
    gold_file = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if file.endswith(".txt")][0]

    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    gold_keys = []
    with open(gold_file, "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()


    buckets = []
    num = 0
    for i in range(len(sentences)):
        for j in range(len(targets_index_start[i])):
            sentence = ' '.join(sentences[i])
            target_start = targets_index_start[i][j]
            target_end = targets_index_end[i][j]
            target_id = targets[i][target_start]
            target_lemma = lemmas[i][target_start]
            target_pos = poss[i][target_start]
            sense_key = gold_keys[num]
            # Store items
            buckets.append(
                {
                    'sentence': sentence,
                    'target_index_start': target_start,
                    'target_index_end': target_end,
                    'target_id': target_id,
                    'target_lemma': target_lemma,
                    'target_pos': target_pos,
                    'sense_key': sense_key
                }
            )
            num += 1
    df = pl.DataFrame(buckets)
    return df.to_pandas()



class DataBuilder(object):
    def __init__(self, dir_path, pos: str = "ALL", seed: Union[int, None]=None) -> None:
        self.dir_path = dir_path
        self.POS_MAP = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ':  wn.ADJ, 'ADV':  wn.ADV}
        self.pos = pos
        self.seed = seed

        if pos not in self.POS_MAP and pos != "ALL":
            allowed = list(self.POS_MAP.keys())
            allowed.append("ALL")
            raise ValueError(f"Invalid POS: {pos}, Should be one of {allowed}")
        assert os.path.exists(dir_path), f"{dir_path} does not exist"
        assert os.path.isdir(dir_path), f"{dir_path} is not a directory"
        assert os.listdir(dir_path), f"{dir_path} is empty"
        

    def __call__(self) -> pl.DataFrame:
        df = generate(self.dir_path)

        df['sense_list'] = df.apply(lambda x: get_all_senses(x, self.POS_MAP), axis=1)
        df['correct_sense'] = df.apply(get_correct_sense, axis=1)
        df = df.apply(fill_empty_sense, axis=1)

        if self.seed is not None:
            random.seed(self.seed) 
            df['sense_list'] = df['sense_list'].apply(lambda x: random.sample(x, len(x)))
        if self.pos == "verb":
            return df[df['target_pos'] == "VERB"].reset_index(drop=True)
        elif self.pos == "noun":
            return df[df['target_pos'] == "NOUN"].reset_index(drop=True)
        elif self.pos == "adj":
            return df[df['target_pos'] == "ADJ"].reset_index(drop=True)
        elif self.pos == "adv":
            return df[df['target_pos'] == "ADV"].reset_index(drop=True)
        else:
            return df.reset_index(drop=True)

    def __repr__(self) -> str:
        return f"Preprocess(dir_path={self.dir_path})"

    def to_pandas(self) -> pl.DataFrame:
        return self()

    def to_csv(self, path: str = None) -> None:
        self().to_csv(
            path or f"{self.dir_path.split('/')[-1]}.csv", 
            index=False, encoding="utf-8"
            )

    def to_json(self, path: str = None) -> None:
        self().to_json(
            path or f"{self.dir_path.split('/')[-1]}.json", 
            orient="records", force_ascii=False
            )


__all__ = ["DataBuilder"]