import pandas as pd
import hazm
import re

import time

CONJUCTIONS = {'-', "،", ":", ".", "•", "!", "؟", "؛", "‌", "(", ")", "»", "«"}
STOP_WORDS = set(hazm.stopwords_list())

norm = hazm.Normalizer()
lem = hazm.Lemmatizer()
tok = hazm.WordTokenizer()

def normalize_text(text):
    text = norm.normalize(text)
    return tok.tokenize(text)

def clean_word(word):
    word = lem.lemmatize(word).split("#")[-1]
    return word

def normalize_row(row):
    row.title = normalize_text(row.title)
    row.description = normalize_text(row.description)
    return row

def lemmatize_row(row):
    row.title = list(map(clean_word, row.title))
    row.description = list(map(clean_word, row.description))
    return row

def filter_row(row):
    row.title = list(filter(is_important, row.title))
    row.description = list(filter(is_important, row.description))
    return row

def is_important(word):
    if re.search(r'\d', word): return False
    if word in CONJUCTIONS: return False
    if word in STOP_WORDS: return False
    return True

def preprocess_df(dataframe):
    dataframe.apply(normalize_row, axis=1)
    dataframe.apply(filter_row, axis=1)
    dataframe.apply(lemmatize_row, axis=1)
