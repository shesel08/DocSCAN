import argparse
import csv
import json
import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer


def preprocess_data(infile, outfile):
    texts = []

    print("Loading the pre-trained SBERT model")
    # Load the pre-trained SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    with open(infile, 'r') as file:
        for line in file:
            item = json.loads(line)  # Parse each line as JSON
            # Assume each line has a "text" field; adjust based on your data's structure
            if "text" in item:
                texts.append(item["text"])
                print("Computing embeddings")
                # Compute embeddings
                embeddings = model.encode(texts)
                out = {"embeddings": embeddings, "label": item["label"]}
                print(out)

    with open(outfile, 'wb') as file:
        pickle.dump(out, file)
        print(f"Embeddings saved to {outfile}")
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="ag_news", type=str, help="")
    args = parser.parse_args()

    if args.experiment == "ag_news":
        preprocess_data('ag_news/train.jsonl', 'ag_news/train_embedded.pkl')
        preprocess_data('ag_news/test.jsonl', 'ag_news/test_embedded.pkl')
    elif args.experiment == "google_snippets":
        preprocess_data('google_snippets/train.jsonl', 'google_snippets/train_embedded.pkl')
        preprocess_data('google_snippets/test.jsonl', 'google_snippets/test_embedded.pkl')
    elif args.experiment == "dbpedia":
        preprocess_data('dbpedia/train.jsonl', 'dbpedia/train_embedded.pkl')
        preprocess_data('dbpedia/test.jsonl', 'dbpedia/test_embedded.pkl')
    else:
        preprocess_data('ag_news/train.jsonl', 'ag_news/train_embedded.pkl')
        preprocess_data('ag_news/test.jsonl', 'ag_news/test_embedded.pkl')
