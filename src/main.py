import pandas as pd
import time

from pre import preprocess_df
from pro import create_bow, predict_cat, get_avg_acc, get_result

BOOKS_TRAIN_PATH = "./books_train.csv"
BOOKS_TEST_PATH = "./books_test.csv"

def main():
    t1 = time.monotonic()
    train_df = pd.read_csv(BOOKS_TRAIN_PATH)
    test_df = pd.read_csv(BOOKS_TEST_PATH)
    t2 = time.monotonic()

    print("Reading CSV:", end=" ")
    print(t2 - t1)
    
    t1 = time.monotonic()
    preprocess_df(train_df)
    preprocess_df(test_df)
    t2 = time.monotonic()

    print("Preprocessing:", end=" ")
    print(t2 - t1)

    t1 = time.monotonic()
    bow = create_bow(train_df)
    t2 = time.monotonic()

    print("Creating BoW:", end=" ")
    print(t2 - t1)

    t1 = time.monotonic()
    prediction = predict_cat(test_df, bow)
    t2 = time.monotonic()

    print("Prediction:", end=" ")
    print(t2 - t1)

    acc = get_avg_acc(test_df, prediction)
    res = get_result(test_df, prediction)

    # for c1 in res:
    #     print("For Category: ", end="")
    #     print(c1)
    #     for c2 in res[c1]:
    #         print("Gussed ", end="")
    #         print(c2, end=" ")
    #         print(res[c1][c2], end=" times\n")

    print("Accuracy: ", end=" ")
    print(acc, end="%\n")

if __name__ == "__main__":
    main()
