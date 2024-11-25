import argparse
import os
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


def evaluate(y, preds):
    print(metrics.classification_report(y, preds))
    print("accuracy", metrics.accuracy_score(y, preds))
    return metrics.classification_report(y, preds), metrics.accuracy_score(y, preds)

def convertToList(df_data, value):
    output = []
    for data in df_data:
        if value in data:
            output.append(data[value])
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="ag_news", help="")
    parser.add_argument("--features", type=str, default="sbert", help="")
    parser.add_argument("--train_file", type=str, default="ag_news/train_embedded.pkl", help="")
    parser.add_argument("--test_file", type=str, default="ag_news/test_embedded.pkl", help="")

    args = parser.parse_args()

    if args.train_file and args.test_file:
        fn_train = args.train_file
        fn_test = args.test_file
    else:
        fn_train = os.path.join(args.path, "train_embedded.pkl")
        fn_test = os.path.join(args.path, "test_embedded.pkl")



    if args.features == "sbert":
        df_train = pd.read_pickle(fn_train)
        df_test = pd.read_pickle(fn_test)
        X_train = convertToList(df_train, "embeddings")
        X_test = convertToList(df_test, "embeddings")
        y_train = convertToList(df_train, "label")
        y_test = convertToList(df_test, "label")
    elif args.features == "tfidf":
        df_train = pd.read_json(fn_train, lines=True)
        df_test = pd.read_json(fn_test, lines=True)
        vect = CountVectorizer(max_df=0.7, min_df=3, ngram_range=(1, 2), lowercase=True, stop_words="english")
        tfidf = TfidfTransformer()
        vectorized_train = vect.fit_transform(df_train["text"])
        X_train = tfidf.fit_transform(vectorized_train)
        vectorized_test = vect.transform(df_test["text"])
        X_test = tfidf.transform(vectorized_test)
        y_train = np.array(df_train["label"])
        y_test = np.array(df_test["label"])

    svm = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-5, random_state=42,
                        max_iter=500, tol=1e-5, class_weight=None, verbose=1)

    svm.fit(X_train, y_train)
    labels = svm.predict(X_test)
    classification_report, acc = evaluate(labels, y_test)

    with open(os.path.join(args.path, "results_svm_" + args.features + ".txt"), "w") as outfile:
        outfile.write(classification_report + "\n")
        outfile.write("ACCURACY" + str(acc))
