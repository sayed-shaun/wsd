import nltk
import polars as pl
from nltk.corpus import wordnet as wn
try:
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading WordNet: {e}")
from typing import Optional, Dict, Any, List, Union


def get_all_senses(row: pl.Series, pos_map: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Return up to `max_senses` for this row's lemma+POS.
    If you have a sense_key and want to enforce it exactly,
    you could insert it first and then fill up to `max_senses`
    with other candidates.
    """
    wn_pos = pos_map.get(row['target_pos'], None)
    if wn_pos:
        syns = wn.synsets(str(row['target_lemma']), pos=wn_pos)
    elif row['target_lemma'] != '' or row['target_lemma'] is not None:
        syns = wn.synsets(str(row['target_lemma']))
    else:
        word = row['sentence'].split()[row['target_index_start']]
        syns = wn.synsets(str(word))

    return [s.definition() for s in syns]


def get_correct_sense(row: pl.Series) -> Union[str, None]:
    """
    Return the correct sense for this row's lemma+POS.
    Args:
        row (pl.Series): A row from a DataFrame containing 'sense_key'.
    Returns:
        Union[str, None]: The definition of the correct sense or None.
    """
    if 'sense_key' in row and row['sense_key'] is not None and row['sense_key'] != '':
        lem = wn.lemma_from_key(row['sense_key'])
        syn = lem.synset()
        return syn.definition()


def fill_empty_sense(row: pl.Series) -> pl.Series:
    """
    Fill empty sense_list with the first sense of the word.
    Args:
        row (pl.Series): A row from a DataFrame containing 'sentence', 
                         'target_index_start', and 'sense_list'.
    Returns:
        pl.Series: The updated row with 'sense_list' filled.
    """
    if not row['sense_list']:
        word = row['sentence'].split()[row['target_index_start']]
        syns = wn.synsets(str(word))
        row['sense_list'] = [s.definition() for s in syns]
    return row