import argparse
import csv
import json
import os


def preprocess_agnews_data():
    path_train = "../data/ag_news_csv/train.csv"
    path_test = "../data/ag_news_csv/test.csv"
    path_out = "../ag_news"

    def generate_agnews_data(path, mode, path_out):
        # Create the directory if it doesn't exist
        os.makedirs(path_out, exist_ok=True)

        # Now open the file
        outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                out = {"text": row[2], "label": int(row[0])}
                json.dump(out, outfile)
                outfile.write("\n")
        outfile.close()

    generate_agnews_data(path_train, "train", path_out)
    generate_agnews_data(path_test, "test", path_out)


def preprocess_google_snippets_data():
    path_train = "../data/google_snippets/data-web-snippets/train.txt"
    path_test = "../data/google_snippets/data-web-snippets/test.txt"
    path_out = "../google_snippets"

    def generate_google_snippets_data(path, mode, path_out):
        # Create the directory if it doesn't exist
        os.makedirs(path_out, exist_ok=True)

        # Now open the file
        outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.strip().split()
                text = " ".join(line[:-1])
                label = line[-1]
                out = {"text": text, "label": label}
                json.dump(out, outfile)
                outfile.write("\n")
        outfile.close()

    generate_google_snippets_data(path_train, "train", path_out)
    generate_google_snippets_data(path_test, "test", path_out)


def preprocess_dbpedia_data():
    path_train = "../data/dbpedia/dbpedia_csv/train.csv"
    path_test = "../data/dbpedia/dbpedia_csv/test.csv"
    path_out = "../dbpedia"

    def generate_dbpedia_data(path, mode, path_out):
        # Create the directory if it doesn't exist
        os.makedirs(path_out, exist_ok=True)

        # Now open the file
        outfile = open(os.path.join(path_out, mode + ".jsonl"), "w")
        X, y = [], []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                X.append(row[-1])
                y.append(row[0])

        for text, label in zip(X, y):
            out = {"text": text, "label": label}
            json.dump(out, outfile)
            outfile.write("\n")
        outfile.close()

    generate_dbpedia_data(path_train, "train", path_out)
    generate_dbpedia_data(path_test, "test", path_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="ag_news", type=str, help="")
    args = parser.parse_args()

    if args.experiment == "ag_news":
        preprocess_agnews_data()
    elif args.experiment == "google_snippets":
        preprocess_google_snippets_data()
    elif args.experiment == "dbpedia":
        preprocess_dbpedia_data()
    else:
        preprocess_agnews_data()
