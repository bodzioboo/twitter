#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:35:12 2020

@author: piotr
"""

import csv
import pandas as pd
import dask.dataframe as dd
import itertools
import os
import datetime as dt
import re
import enchant
import stanza
from nltk.metrics.distance import edit_distance
import logging
from tqdm import tqdm
import pdb
import numpy as np
from itertools import islice, repeat, chain
import gc
from operator import itemgetter
from functools import partial

# use logger of whatever file it's executed from
logger = logging.getLogger(__name__)
logger.propagate = False


def csv_bind(file1, file2, check_headers=True):
    if check_headers:
        # make sure that the column headings are the same
        f1cols = os.popen(f"head -1 {file1}").read()
        f2cols = os.popen(f"head -1 {file2}").read()
        assert (f1cols == f2cols)

    with open(file2, "r") as f2:
        reader = csv.DictReader(f2)
        next(reader)

        for _ in itertools.count():
            data = list(itertools.islice(reader, 100000))

            if not data:
                break

            with open(file1, "a") as f1:
                writer = csv.DictWriter(f1, fieldnames=data[0].keys())
                writer.writerows(data)
            logger.info("Appended new rows")


def drop_date(path_source, path_target, date):
    with open(path_source, "r") as f1:
        reader = csv.DictReader(f1)

        for _ in itertools.count():

            check = list(itertools.islice(reader, 100000))
            if not check:
                break

            fmt = "%a %b %d %H:%M:%S +0000 %Y"
            condition = dt.datetime.strptime(date, "%Y/%m/%d")
            # date is not text:
            redate = "[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \+\d{4} \d{4}"
            # pdb.set_trace()
            check = [elem for elem in check if re.match(redate, str(elem["created_at"]))]
            check = [elem for elem in check if dt.datetime.strptime(elem["created_at"], fmt) < condition]

            length_out = len(check)

            if check:

                write_mode = "a" if os.path.isfile(path_target) else "w"
                with open(path_target, write_mode) as f2:
                    fieldnames = list(check[0].keys())
                    writer = csv.DictWriter(f2, fieldnames=fieldnames)
                    if write_mode == "w":
                        writer.writeheader()
                    writer.writerows(check)
                logger.info(f"Wrote {length_out} to file {path_target}")


def tweet_stats(twitter_text):
    """
    Function for counting stuff in tweets. Takes in a list, returns a dict.
    
    """

    tweets = list(twitter_text)

    counts = dict()

    # count URls
    re_url = re.compile(r'(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:\/\S+)*)?')
    urls = [re_url.findall(x) for x in tweets]
    counts["urls"] = [len(x) for x in urls]

    # remove urls (so they don't interfere with other counts)
    tweets = [tweet.lower() for tweet in tweets]  # lowercase
    tweets = [
        re.sub(r'(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:\/\S+)*)?', ' ',
               tweet) for tweet in tweets]
    # remove extra whitespaces:
    tweets = [re.sub(r' +', ' ', tweet) for tweet in tweets]

    # get counts of emojis and emoticons
    re_emoji = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    emojis = [re_emoji.findall(x) for x in tweets]
    counts["emoti"] = [len(x) for x in emojis]

    re_emoticons_pos = re.compile(r"[oO>]?[;:=Xx8]+'?\-?[)>)pPdD3\]\}\*]+")
    emoticons_pos = [re_emoticons_pos.findall(x) for x in tweets]
    counts["emoti_pos"] = [len(x) for x in emoticons_pos]

    re_emoticons_neg = re.compile(r"[oO>]?[:;=]+'?\-?[(<\\\/\[(\{]+")
    emoticons_neg = [re_emoticons_neg.findall(x) for x in tweets]
    counts["emoti_neg"] = [len(x) for x in emoticons_neg]

    re_emoticons_neut = re.compile(r"[oO>]?[:;=]+'?\-?[oO]+")
    emoticons_neut = [re_emoticons_neut.findall(x) for x in tweets]
    counts["emoti_neut"] = [len(x) for x in emoticons_neut]

    # get hashtags counts
    re_hashtag = re.compile(r'\#[a-zA-Z0-9]+')
    hashtags = [re_hashtag.findall(x) for x in tweets]
    counts["hashtags"] = [len(x) for x in hashtags]

    # get mention counts
    re_mention = re.compile(r'\@\w+(?!\.\w+)\b')
    mentions = [re_mention.findall(x) for x in tweets]
    counts["mentions"] = [len(x) for x in mentions]

    # get number counts
    re_number = re.compile(r"(\S+)?\d+(\S+)?")
    numbers = [re_number.findall(x) for x in tweets]
    counts["numbers"] = [len(x) for x in numbers]

    return counts


def clean_tweets(text,
                 hashtag_replace=False,
                 url_token=False,
                 emoji_token=False,
                 numbers_token=False,
                 hashtag_token=False,
                 mention_token=False):
    # store tokens for replacement:

    url = emoji = emoticon_pos = emoticon_neg = emoticon_neut = numbers = hashtag = mention = ' '

    if url_token:
        url = " urltoken "

    if emoji_token:
        emoji = " emojitoken "
        emoticon_pos = " emoticon_pos "
        emoticon_neg = " emoticon_neg "
        emoticon_neut = " emoticon_neut "

    if numbers_token:
        numbers = " numbertoken "

    if hashtag_token:
        hashtag = " hashtagtoken "

    if mention_token:
        mention = " mentiontoken "

    # urls
    url_regex = r"(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:(\/|\?)\S+)*)?"
    text = re.sub(url_regex, url, text)

    # hashtags - this is done before lowercasing, as I attempt to
    # separate hashtags into specific tokens, e.g. #DonaldTrump -> Donald Trump
    if hashtag_replace:
        text = re.sub(r'\#[a-zA-Z0-9]+', hashtag, text)
    else:
        if re.search("#", text):
            text = re.sub("#", " ", text)  # replace # with space
            text = re.sub(r"([a-z])([A-Z])", r'\1 \2', text)  # insert spaces on capital letters
            text = re.sub(r"([A-Za-z])([0-9])", r'\1 \2', text)  # insert spaces on word-number boundaries

    # emoticons and emojis
    re_emoji = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+")

    text = re.sub(re_emoji, emoji, text)
    text = re.sub(r"[oO>]?[;:=Xx8]+'?\-?[)>)pPdD3\]\}\*]+", emoticon_pos, text)
    text = re.sub(r"[oO>]?[:;=]+'?\-?[(<\\\/\[)\{]+", emoticon_neg, text)
    text = re.sub(r"[oO>]?[:;=]+'?\-?[oO]+", emoticon_neut, text)

    # remove words of length == 1
    text = re.sub(r'\b(\w)\b', ' ', text)

    # digits
    text = re.sub(r"(\S+)?\d+(\S+)?", numbers, text)

    # rt flags
    text = re.sub(r"^rt", ' ', text)

    # newline
    text = re.sub("\\n", ' ', text)

    # usermentions
    text = re.sub(r'\@\w+(?!\.\w+)\b', mention, text)

    # non alphanumeric
    text = re.sub(r"\&amp", " ", text)  # bug in twitter - &amp appearing
    text = re.sub(r"\W+", " ", text)
    # tweets = [re.sub(r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\…\–]+"," ",tweet) for tweet in tweets]
    # underscores
    text = re.sub(r"_+", " ", text)

    # remove extra whitespaces:
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    return text


def tag_retweets(df):
    """
    Mark retweets in data scraped from twitter

    Parameters
    ----------
    df : pd.DataFrame.

    Returns
    -------
    df : pd.DataFrame.

    """
    df["retweet"] = df["full_text"].str.startswith("RT")
    df.loc[df.retweet, 'full_text'] = df.loc[df.retweet, 'retweeted_status-full_text']
    return df


def check_errors(df):
    """

    Cleaning the scraped data
    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_good : pd.DataFrame
        Dictionary with dropped corrupt records.

    """

    if df.empty:
        return df

    len_in = df.shape[0]

    re_date = "[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \+\d{4} \d{4}"
    index_good = df["created_at"].str.match(re_date)

    should_be_numeric = ["id_str", "user-id_str"]
    index_good &= df[should_be_numeric].applymap(lambda x: str(x).isnumeric()).all(axis=1)

    df_good = df[index_good]
    len_out = df_good.shape[0]

    logger.info(f"Number of corrupt records {len_in - len_out}")

    return df_good


class SpellChecker:
    def __init__(self, threshold=0.3):
        self.dict_pl = enchant.Dict("pl_PL")
        self.dict_en = enchant.Dict("en_EN")
        self.nlp_en = stanza.Pipeline("en")
        self.threshold = threshold

    def check(self, texts: list):
        """
        Core function
        """

        for i, text in enumerate(texts):
            texts[i] = [self.language_check(word) for word in text]

    def language_check(self, word: str):
        """
        Verify the language of the word, if not English or Polish
        find the best alternative. 
        """

        if self.dict_pl.check(word):
            return word
        elif self.dict_en.check(word):
            return self.nlp_en(word).sentences[0].words[0].lemma
        else:
            return self.best_match(word)

    def best_match(self, text: str):
        """
        If word not in dictionary, return best match under threshold
        of edit distance to length ratio
        """

        if not text:
            return None

        pl = self.dict_pl.suggest(text)
        en = self.dict_en.suggest(text)

        dist_en = None
        dist_pl = None

        if en and edit_distance(en[0], text) / len(text) < self.threshold:
            en = en[0]
            dist_en = edit_distance(en[0], text)

        if pl and edit_distance(pl[0], text) / len(text) < self.threshold:
            pl = pl[0]
            dist_pl = edit_distance(pl, text)

        if dist_pl and dist_en:
            if dist_pl > dist_en:
                return en
            else:
                return pl
        elif dist_en:
            return en
        elif dist_pl:
            return pl
        else:
            return None


def filter_date(name: str, start: str, end: str,
                regex: str = '\d{4}\_\d{2}\_\d{2}', fmt: str = '%Y_%m_%d'):
    date = re.search(regex, name).group(0)
    date, start, end = map(lambda x: dt.datetime.strptime(x, fmt), [date, start, end])
    return date >= start and date <= end


def read_dask(files: list, regex: str, filter_fun=None, **kwargs):
    """
    Returns dask array given a list of files.
    """
    ddf = dd.read_csv(files, include_path_column='source', **kwargs)
    if 'Unnamed: 0' in ddf.columns:
        ddf = ddf.drop(columns=['Unnamed: 0'])
    ddf[['source']] = ddf['source'].str.extract(regex, expand=True)
    if filter_fun is not None:
        ddf = filter_fun(ddf)
    return ddf


def read_pandas(files: list, regex: str,
                filter_fun=None,
                batch_size: int = 1, **kwargs):
    """
    Returns pandas iterator given a list of file.

    Parameters
    ----------
    files : list
        list of paths to files.
    ndays : int
        number of days to load.
    filter_fun : function, optional
        function to apply to the data. The default is None.
    batch_size : int, optional
        Batch size for the iterator. The default is 1 (day).
    dtype : str, optional
        Argument for pd.read_csv. The default is None.

    Yields
    ------
    data : pd.DataFrame
        dataframe for both sources for a given batch size.

    """
    prefixes = list(set([re.search(regex, f).group(0) for f in files]))
    ndays = int(len(files) / len(prefixes))
    for _ in range(ndays // batch_size):
        data = pd.DataFrame()
        for prefix in prefixes:
            n_newest = [elem for elem in files if prefix in elem][:batch_size]
            for file in n_newest:
                ind = files.index(file)
                fname = files.pop(ind)  # get filename
                tmp = pd.read_csv(fname, index_col=0, **kwargs)
                tmp["source"] = prefix
                data = data.append(tmp)
        data.reset_index(inplace=True, drop=True)
        if filter_fun:
            data = filter_fun(data)
        yield data


def read_files(path: str, day_from: str, day_to: str, dtype: str = None,
               filter_fun=None, regex: str = r'([a-z]+)(?=\_tweets)',
               method: str = 'pandas', batch_size: int = 1, **kwargs):
    """
    

    Parameters
    ----------
    path : str
        path to folder containing files.
    day_from : str
        date from which to load data %Y_%m_%d.
    day_to : str
        date to which to load data %Y_%m_%d.
    dtype : str, optional
        DESCRIPTION. The default is None.
    filter_fun : TYPE, optional
        DESCRIPTION. The default is None.
    regex : str, optional
        DESCRIPTION. The default is r'([a-z]+)(?=\_tweets)'.
    method : str, optional
        DESCRIPTION. The default is 'pandas'.
    batch_size : int, optional
        DESCRIPTION. The default is 1.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if not os.path.exists(path):
        raise ValueError("Incorrect path")

    files = os.listdir(path)
    if not files:
        raise ValueError("Empty path")

    pattern = r'[a-z]{3}_tweets_\d{4}\_\d{2}\_\d{2}\.csv'
    files = sorted([f for f in files if re.fullmatch(pattern, f)])
    files = [os.path.join(path, f) for f in files if filter_date(f, start=day_from, end=day_to)]

    if method == 'pandas':
        return read_pandas(files=files, regex=regex, filter_fun=filter_fun, dtype=dtype,
                           batch_size=batch_size, **kwargs)
    elif method == 'dask':
        return read_dask(files, regex=regex, filter_fun=filter_fun, dtype=dtype, **kwargs)


def read_window(path, n=7, **kwargs):
    reader = read_files(path=path, method='pandas',**kwargs)
    result = tuple(islice(reader, n))
    if len(result) == n:
        yield pd.concat(result)
    for elem in reader:
        result = result[1:] + (elem,)
        yield pd.concat(result)


def filter_data(path_source: str, path_target: str, nfiles: int, filter_fun, **kwargs):
    """
    Extension of the function above that writes the files to one csv

    Parameters
    ----------
    path_source : str
        source directory.
    path_target : str
        target file.
    nfiles : int
        number of file pairs.
    filter_fun : function
        function applied to data frame.

    Returns
    -------
    None.

    """
    # clean:
    if os.path.isfile(path_target):
        os.remove(path_target)

    for data in tqdm(read_files(path_source, nfiles, prefixes=["gov", "opp"],
                                dtype=str,
                                filter_fun=filter_fun, **kwargs)):
        if os.path.isfile(path_target):
            data.to_csv(path_target, mode="a", header=False)
        else:
            data.to_csv(path_target, mode="w")


def batch(iterable, n=1):
    try:
        l = len(iterable)
    except:
        try:
            l = iterable.shape[0]
        except:
            raise (TypeError('Wrong iterable'))
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def tweets_datesplit(path_source: str, dir_target: str = None, chunksize: int = 100000):
    dir_source, file_source = os.path.split(path_source)

    if dir_target is None:
        dir_target = dir_source

    for df in tqdm(pd.read_csv(path_source, index_col=0,
                               dtype=str, iterator=True, error_bad_lines=False, chunksize=chunksize)):

        df = check_errors(df)

        date = pd.to_datetime(df['created_at']).dt.date

        for day, df in df.groupby(date):

            file_target = file_source.split('.')[0] + '_' + day.strftime('%Y_%m_%d') + '.csv'

            path_target = os.path.join(dir_target, file_target)

            if os.path.isfile(path_target):

                df.to_csv(path_target, mode='a', header=False)

            else:

                df.to_csv(path_target)


class VectorDict:
    def __init__(self):
        self.matrix = None
        self.keys = None

    def _load(self, file, dim: int, vocab: dict = None, dtype=np.float32):
        """
        Load vectors to class
        """
        mapping = dict()
        for line in file:
            vec = line.split()
            if len(vec) == dim + 1:
                mapping[vec[0]] = [float(elem) for elem in vec[1:]]
        self.matrix = np.vstack(list(mapping.values()))
        self.keys = np.array(list(mapping.keys()))
        del mapping
        gc.collect()
        return self

    def read_file(self, path: str, dim: int, **kwargs):
        """
        Read vectors from text file.
        path: str
            path to vec file
        dim: int
            Dimensionality of the word vector
        **kwargs: arguments for _load
        
        """
        with open(path, 'r') as f:
            return self._load(f, dim, **kwargs)

    def __getitem__(self, key: str):
        if key not in self.keys:
            raise (KeyError('Not a valid key'))
        ind = np.where(self.keys == key)[0]
        return self.matrix[ind].flatten()

    def _cosine_sim(self, vec1: np.array, vec2: np.array):
        dots = vec1 @ vec2.T
        norms = np.outer(np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1))
        return dots / norms

    def most_similar_by_vectors(self, vector: np.array, n: int = 1):
        comp = self._cosine_sim(self.matrix, vector)  # get pairwise cosine similarity
        indices = (-comp).argsort(axis=0)[:n]  # get indices of top n for each of the vectors
        result = [list(zip(self.keys[ind].tolist(), comp[ind, i].tolist())) for i, ind in enumerate(indices.T)]
        return result

    def most_similar_by_words(self, words: list, n: int = 1):
        vec = np.array(itemgetter(*words)(self))
        if len(vec.shape) != 2:
            vec = vec[np.newaxis]
        result = self.most_similar_by_vectors(vec, n=n)
        return result


class VectorDictLowMemory(VectorDict):
    def __init__(self, path, dim, chunk_size: int = 100000):
        self.path = path
        self.chunk_size = chunk_size
        self.dim = dim

    def read_in_chunks(self, connection, chunk_size: int):
        for _ in tqdm(repeat(0)):
            data = list(islice(connection, chunk_size))
            if not data:
                break
            yield data

    def apply_in_chunks(self, func, **kwargs):
        """Lazy function (generator) to read a file piece by piece."""
        result = []
        for data in self.read_in_chunks(connection=open(self.path, 'r'), chunk_size=self.chunk_size):
            super()._load(data, dim=self.dim)
            res = func(**kwargs)
            result.append(res)
        return result

    def getter(self, keys: list):
        """
        Get word vectors given a string.

        Parameters
        ----------
        key : list
            words.

        Returns
        -------
        np.array
            word vector

        """
        vectors = []
        for data in self.read_in_chunks(open(self.path, 'r'), self.chunk_size):
            super()._load(data, dim=self.dim)
            for key in keys:
                if key in self.keys:
                    vectors.append(self[key])
                else:
                    continue
            if len(vectors) == len(keys):
                return np.array(vectors)
        raise KeyError('one of the keys not found')

    def most_similar_by_vectors(self, vector: np.array, n: int = 1):
        """
        Given a vector or array of vector  returns n most similar vectors for each.
        ----------
        vector : np.array
            (N x dim) where dim is the dimensionality of the embeddings
        n : int, optional
            number of words to get.. The default is 1.

        Returns
        -------
        results : list of lists of (word, similarity_score) tuples.

        """
        results = self.apply_in_chunks(super().most_similar_by_vectors, vector=vector, n=n)
        results = np.array(results).T.tolist()  # transpose
        results = [chain.from_iterable(res) for res in results]  # join all lists
        results = [sorted(res, key=lambda x: x[1], reverse=True)[:n] for res in results]  # get top n
        return results

    def most_similar_by_words(self, word: list, n: int = 1):
        """
        Given a list of words, return n most similar words for each.

        Parameters
        ----------
        word : list
            list of strings.
        n : int, optional
            number of words to get. The default is 1.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """
        vectors = self.getter(word)
        results = self.most_similar_by_vectors(vectors, n)
        return results


if __name__ == "__main__":
    pass
