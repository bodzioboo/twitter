#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:19:45 2020

@author: piotr
"""
import stanza
import os
import pandas as pd
import enchant
from tqdm import tqdm
from .utils import clean_tweets, tag_retweets, check_errors
import json
import collections
import argparse
import logging
import pdb

logger = logging.getLogger("__main__")


class Lemmatizer:
    def __init__(self, stopwords=None, use_gpu=True,
                 processors="tokenize,mwt,pos,lemma", **kwargs):

        self.nlp_pl = stanza.Pipeline("pl", processors=processors,
                                      use_gpu=use_gpu, tokenize_no_ssplit=True,
                                      **kwargs)
        self.logger = logging.getLogger(__name__)

        if stopwords:
            self.stopwords = []
            with open(stopwords, "r") as f:
                for line in f:
                    self.stopwords.append(line.strip("\n"))
        else:
            self.stopwords = []

    def lemmatize(self, texts: list):
        """

        :param texts: list of sentences
        :return: lemmas, tokens
        """

        doc = self.bind(texts)
        doc = self.nlp_pl(doc)

        tokens = []
        lemmas = []
        for sent in doc.sentences:
            lemmas.append([word.lemma for word in sent.words if word.lemma not in self.stopwords])
            tokens.append([word.text for word in sent.words])
        self.logger.info(f"Lemmatized {len(lemmas)} sentences")
        return lemmas, tokens

    def bind(self, texts: list):
        # Bind texts for stanza pipeline
        return "\n\n".join(texts)


class LanguageTagger:

    def __init__(self, lang="pl"):
        self.dict = enchant.Dict(lang)

    def tag(self, sentence: str):
        sent = sentence.split()
        checks = [self.dict.check(word) for word in sent]
        return sum(checks) / len(sent)


class Preprocessor:
    def __init__(self, keep_cols=None, chunksize=100000, record_path=None, **kwargs):

        self.cols = ["full_text", "created_at", "id_str", "user-id_str", "retweet"]

        if keep_cols:
            self.cols = list(set(self.cols + keep_cols))

        self.chunksize = chunksize  # chunk size for loader

        self.lem = Lemmatizer(**kwargs)

        self.tagger = LanguageTagger()

    def preprocess(self, df):

        df = check_errors(df)  # error check (column format etc)

        df = df[df.lang.isin(['pl', 'und'])]  # filter Polish/undefined language

        if df.empty:
            return None

        df = tag_retweets(df)  # tag retweets

        df = df[self.cols]  # keep important columns

        # clean mentions, interp, etc.:
        df["preprocessed"] = df["full_text"].astype(str).apply(lambda x: clean_tweets(x))

        # drop empty rows before lemmatizing:
        indx = (df["preprocessed"] != "") & (df["preprocessed"].notna())
        df = df[indx]

        # lemmatize - use a dictionary of unique tweets to save time
        tweet_dict = dict().fromkeys(df["preprocessed"].tolist())
        lemmas, tokens = self.lem.lemmatize(list(tweet_dict))  # lemmatization
        lemma_dict = {k: " ".join(lemmas[i]) for i, k in enumerate(tweet_dict)}
        token_dict = {k: " ".join(tokens[i]) for i, k in enumerate(tweet_dict)}
        df["tokenized"] = df["preprocessed"].map(token_dict)  # lemmatized and without stopwords
        df["lemmatized"] = df["preprocessed"].map(lemma_dict)  # just tokenized
        df[["tokenized", "lemmatized"]] = df[["tokenized", "lemmatized"]].applymap(lambda x: x.lower())

        # count proportion of Polish vocabulary
        df["polish"] = df["lemmatized"].apply(self.tagger.tag)

        # drop
        df.drop(columns=["preprocessed"], inplace=True)

        # get day
        df["day"] = pd.to_datetime(df["created_at"],
                                   format="%a %b %d %H:%M:%S +0000 %Y").dt.date  # get day

        df = df.astype(str)

        # write to file
        return df


if __name__ == "__main__":
    # setup logging:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # setup argument parsing:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to source file",
                        type=str)
    parser.add_argument("target",
                        help="Folder for output if split date True or file if split dates False",
                        type=str)
    parser.add_argument('--split_dates', dest='split', action='store_true')
    parser.add_argument('--one_file', dest='split', action='store_false')
    parser.set_defaults(split=True)
    parser.add_argument("--keep_cols",
                        help="List of columns to keep in the output")
    parser.add_argument("--stopwords", help="Path to stopwords")
    parser.add_argument("--record_path", help="Path to json record")
    parser.add_argument("--chunk_size", help="Chunk size",
                        nargs="?", const=10000, type=int)
    parser.add_argument("--pos_batch_size", help="Batch size for stanza pos tagger",
                        nargs="?", const=1000, type=int)
    args = parser.parse_args()  # parse arguments

    # can provide a path to json file with columns to keep
    if type(args.keep_cols) == str and "json" in args.keep_cols:
        args.keep_cols = json.load(open(args.keep_cols, "r"))

    if args.record_path and os.path.isfile(args.record_path):
        record = json.load(open(args.record_path, "r"))
        record = collections.defaultdict(lambda: 0, record)
    else:
        record = collections.defaultdict(lambda: 0)

    # init the class
    preprocessor = Preprocessor(keep_cols=args.keep_cols, pos_batch_size=args.pos_batch_size)  # init
    names = pd.read_csv(args.source, dtype=str, nrows=1, index_col=0)  # get column names
    names = names.columns

    # read file by chunk:
    for df in tqdm(pd.read_csv(args.source, iterator=True, dtype=str,
                               skiprows=record[args.source] + 1,
                               chunksize=args.chunk_size, index_col=0,
                               names=names)):
        res = preprocessor.preprocess(df)
        if args.split:
            for day, subdf in res.groupby(res["day"]):  # split by day

                if subdf.full_text.str.isnumeric().sum() > subdf.shape[0] / 10:
                    logger.error("Something wrong with columns")
                    continue

                # get target directory:
                target = os.path.split(args.source)[-1].split(".")[0]
                target += "_" + day.replace("-", "_") + ".csv"
                target = os.path.join(args.target, target)

                if os.path.isfile(target):
                    subdf.to_csv(target, mode="a", header=False)  # append if exists

                else:
                    subdf.to_csv(target, mode="w")  # else write to new

                logger.info(f"Wrote {subdf.shape[0]} rows to file {target}")
        else:
            # without splitting by date:
            if os.path.isfile(args.target):
                res.to_csv(args.target, mode="a", header=False)  # append if exists
            else:
                res.to_csv(args.target, mode="w")  # else write to new

        record[args.source] += args.chunk_size

        if args.record_path:
            json.dump(dict(record), open(args.record_path, "w"))
