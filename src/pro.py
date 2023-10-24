import pandas as pd
import math
import time

CATEGORIES = ["مدیریت و کسب و کار", "رمان", "کلیات اسلام", "داستان کودک و نوجوانان", "جامعه‌شناسی", "داستان کوتاه"]

ALPHA = 1
WEIGHT = 5

def create_bow(dataframe):
    bow = dict()
    for c in CATEGORIES:
        bow[c] = dict()

    for _, book in dataframe.iterrows():
        for word in book.title:
            if not word in bow[CATEGORIES[0]]:
                for c in CATEGORIES:
                    bow[c][word] = 0
            bow[book.categories][word] += WEIGHT

        for word in book.description:
            if not word in bow[CATEGORIES[0]]:
                for c in CATEGORIES:
                    bow[c][word] = 0
            bow[book.categories][word] += 1

    return bow

def prob_word_if_cat(bow, word, category, dot_product):
    if word in bow[CATEGORIES[0]]:
        n_w = bow[category][word]
        if n_w == 0:
            return ALPHA / (dot_product[category] + ALPHA * len(bow[CATEGORIES[0]]))
        return n_w / dot_product[category]
    else:
        return ALPHA / (dot_product[category] + ALPHA * len(bow[CATEGORIES[0]]))

def prob_cat_if_book(bow, book, category, dot_product):
    p = 0
    for word in book.title + book.description:
        p += math.log(prob_word_if_cat(bow, word, category, dot_product))

    return p

def get_dot(bow):
    ans = dict()
    for c in bow:
        ans[c] = sum(bow[c].values())
    return ans

def predict_cat(test_df, bow):
    ans = []
    get_dot
    dot_product = get_dot(bow)
    category = CATEGORIES[0]
    for _, book in test_df.iterrows():
        cur_guess = -1e15
        for c in CATEGORIES:
            prob = prob_cat_if_book(bow, book, c, dot_product)
            if prob > cur_guess:
                cur_guess = prob
                category = c;
        ans.append(category)
    return ans

def get_result(test_df, pred):
    res = dict()
    for c1 in CATEGORIES:
        res[c1] = dict()
        for c2 in CATEGORIES:
            res[c1][c2] = 0 

    for i, book in test_df.iterrows():
        res[book.categories][pred[i]] += 1

    return res

def get_avg_acc(test_df, pred):
    s, t = 0, 0
    for i, book in test_df.iterrows():
        t += 1
        if pred[i] == book.categories: s += 1

    return s / t * 100
