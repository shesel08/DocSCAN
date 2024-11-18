import os
import argparse
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy.lang.en import English
from collections import Counter
import numpy as np
from random import randint
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def load_clusters(fn):
    with open(fn) as f:
        clusters = [i.strip() for i in f]
    return clusters


def generate_word_clouds(topic, df_topic, nlp, outpath, vectorizer=None):
    if vectorizer is not None:
        vecs = vectorizer.transform(df_topic["sentence"])
        feature_names = vectorizer.get_feature_names()
        dense = vecs.todense().tolist()
        df_tfidf = pd.DataFrame(dense, columns=feature_names)
        maincol = randint(0, 360)

        def colorfunc(word=None, font_size=None,
                      position=None, orientation=None,
                      font_path=None, random_state=None):
            color = randint(maincol - 10, maincol + 10)
            if color < 0:
                color = 360 + color
            return "hsl(%d, %d%%, %d%%)" % (color, randint(65, 75) + font_size / 7, randint(35, 45) - font_size / 10)

        wordcloud = WordCloud(background_color="white",
                              ranks_only=False,
                              max_font_size=120,
                              color_func=colorfunc,
                              height=600, width=800).generate_from_frequencies(df_tfidf.T.sum(axis=1))
    else:
        word_counts = Counter()
        for sent in df_topic["sentence"]:
            word_counts.update(
                [re.sub("\W*", "", w.lemma_.lower()) for w in nlp(sent) if not w.is_stop and re.sub("\W*", "", w.text)])

        maincol = randint(0, 360)

        def colorfunc(word=None, font_size=None,
                      position=None, orientation=None,
                      font_path=None, random_state=None):
            color = randint(maincol - 10, maincol + 10)
            if color < 0:
                color = 360 + color
            return "hsl(%d, %d%%, %d%%)" % (color, randint(65, 75) + font_size / 7, randint(35, 45) - font_size / 10)

        wordcloud = WordCloud(background_color="white",
                              ranks_only=False,
                              max_font_size=120,
                              color_func=colorfunc,
                              height=600, width=800).generate_from_frequencies(word_counts)
    # print (word_counts.most_common(n=25))

    topic = str(topic)
    if vectorizer is not None:
        # I don't know what I am doing, but this retrieves column names of five highest tf-idf values
        row = df_tfidf.T.sum(axis=1).nlargest()
        # print (row)
        names, _ = list(zip(*row.items()))
        # print (names)
        # names, _ = list(zip(*df_tfidf.T.sum(axis=1).nlargest().items()))
        topic = str(topic)
        if len(topic) == 1:
            topic = "0" + topic
        save_filename = topic + "_" + "_".join(names)
    else:
        # print (topic)
        # if we don't have topic label, but just e.g. cluster "86", we take the first five words as a description
        if len(topic) == 1:
            topic = "0" + topic
        save_filename = topic + "_" + "_".join([i[0] for i in word_counts.most_common(n=5)])

    plt.clf()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    # print (save_filename)
    plt.savefig(os.path.join(outpath, save_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="20newsgroup", type=str, help="", required=True)
    parser.add_argument("--outpath", default="wordclouds", type=str, help="")
    parser.add_argument("--frequencies", default="tf-idf", type=str, help="")

    args = parser.parse_args()

    outpath = os.path.join(args.path, args.outpath)
    os.makedirs(outpath, exist_ok=True)

    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    if os.path.exists(os.path.join(args.path, "test_embedded.pkl")):
        filename = os.path.join(args.path, "test_embedded.pkl")
    else:
        filename = os.path.join(args.path, "train_embedded.pkl")

    df = pd.read_pickle(filename)
    clusters = load_clusters(os.path.join(args.path, "predictions.txt"))
    df["clusters"] = clusters

    if args.frequencies == "tf-idf":
        vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.75, max_features=10000)
        vectorizer.fit(df["sentence"])

    for topic in tqdm(np.unique(clusters)):
        try:
            df_topic = df[df["clusters"] == topic]
            # print (topic, len(df_topic))
            if args.frequencies == "tf-idf":
                generate_word_clouds(topic, df_topic, nlp, outpath, vectorizer)
            else:
                generate_word_clouds(topic, df_topic, nlp, outpath)
        except:
            pass
