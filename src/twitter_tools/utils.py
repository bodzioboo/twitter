#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:35:12 2020

@author: piotr
"""

import csv
import pandas as pd
import itertools
import os
import datetime
import re
import enchant
import stanza
from nltk.metrics.distance import edit_distance
import logging
from tqdm import tqdm
import pdb

#use logger of whatever file it's executed from
logger = logging.getLogger(__name__)
logger.propagate = False

def csv_bind(file1, file2, check_headers = True):
    
    if check_headers:
        #make sure that the column headings are the same
        f1cols = os.popen(f"head -1 {file1}").read()
        f2cols = os.popen(f"head -1 {file2}").read()
        assert(f1cols == f2cols)
    
    with open(file2, "r") as f2:
        reader = csv.DictReader(f2)
        next(reader)
    
        for _ in itertools.count():
            data = list(itertools.islice(reader, 100000))
            
            if not data:
                break
            
            with open(file1, "a") as f1:
                writer = csv.DictWriter(f1, fieldnames = data[0].keys())
                writer.writerows(data)
            logger.info("Appended new rows")
            
            


def drop_date(path_source, path_target, date):
    
    
        
    with open(path_source,"r") as f1:
        reader = csv.DictReader(f1)
        
        
        for _ in itertools.count():
            
            
            check = list(itertools.islice(reader, 100000))                 
            if not check:
                break
                
                
            fmt = "%a %b %d %H:%M:%S +0000 %Y"
            condition = datetime.datetime.strptime(date, "%Y/%m/%d")
                    #date is not text:
            redate = "[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \+\d{4} \d{4}"
            #pdb.set_trace()
            check = [elem for elem in check if re.match(redate, str(elem["created_at"]))]
            check = [elem for elem in check if datetime.datetime.strptime(elem["created_at"],fmt) < condition]
            
            length_out = len(check)
            
        
            if check:
            
                write_mode = "a" if os.path.isfile(path_target) else "w"
                with open(path_target,write_mode) as f2:
                    fieldnames = list(check[0].keys())
                    writer = csv.DictWriter(f2, fieldnames = fieldnames)
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
    
    #count URls
    re_url = re.compile(r'(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:\/\S+)*)?')
    urls = [re_url.findall(x) for x in tweets]
    counts["urls"] = [len(x) for x in urls]
    
    #remove urls (so they don't interfere with other counts)
    tweets = [tweet.lower() for tweet in tweets] #lowercase
    tweets = [re.sub(r'(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:\/\S+)*)?',' ',tweet) for tweet in tweets]
    #remove extra whitespaces:
    tweets = [re.sub(r' +',' ',tweet) for tweet in tweets]
    
    #get counts of emojis and emoticons
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
    
    #get hashtags counts
    re_hashtag = re.compile(r'\#[a-zA-Z0-9]+')
    hashtags = [re_hashtag.findall(x) for x in tweets]
    counts["hashtags"] = [len(x) for x in hashtags]
    
    #get mention counts
    re_mention = re.compile(r'\@\w+(?!\.\w+)\b')
    mentions = [re_mention.findall(x) for x in tweets]
    counts["mentions"] = [len(x) for x in mentions]
    
    #get number counts
    re_number = re.compile(r"(\S+)?\d+(\S+)?")
    numbers = [re_number.findall(x) for x in tweets]
    counts["numbers"] = [len(x) for x in numbers]
    
    return counts





def clean_tweets(text, 
                      hashtag_replace = False, 
                      url_token = False, 
                      emoji_token = False, 
                      numbers_token = False, 
                      hashtag_token = False, 
                      mention_token = False):
    
    #store tokens for replacement:
    
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
    
    #urls
    url_regex = r"(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:(\/|\?)\S+)*)?"
    text = re.sub(url_regex, url ,text)

    
    #hashtags - this is done before lowercasing, as I attempt to 
    #separate hashtags into specific tokens, e.g. #DonaldTrump -> Donald Trump
    if hashtag_replace:
        text = re.sub(r'\#[a-zA-Z0-9]+', hashtag, text)
    else:
        if re.search("#", text):
                text = re.sub("#"," ", text)  #replace # with space
                text = re.sub(r"([a-z])([A-Z])", r'\1 \2', text)  #insert spaces on capital letters
                text = re.sub(r"([A-Za-z])([0-9])", r'\1 \2', text)  #insert spaces on word-number boundaries
    
   
        
    #emoticons and emojis
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
    
    
    #remove words of length == 1
    text = re.sub(r'\b(\w)\b',' ',text)
    
    #digits
    text = re.sub(r"(\S+)?\d+(\S+)?", numbers, text)
        
        
    #rt flags
    text = re.sub(r"^rt",' ', text)
    
    #newline
    text = re.sub("\\n",' ', text)
    
    
    #usermentions
    text = re.sub(r'\@\w+(?!\.\w+)\b', mention, text)
    
    #non alphanumeric
    text = re.sub(r"\&amp"," ",text) #bug in twitter - &amp appearing
    text = re.sub(r"\W+"," ",text) 
    #tweets = [re.sub(r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\…\–]+"," ",tweet) for tweet in tweets]
    #underscores
    text = re.sub(r"_+"," ",text)
    
    
    #remove extra whitespaces:
    text = re.sub(r' +',' ',text)
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
    df.loc[df.retweet,'full_text'] = df.loc[df.retweet,'retweeted_status-full_text']
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
    

    should_be_numeric = ["id_str","user-id_str"]
    index_good &= df[should_be_numeric].applymap(lambda x: str(x).isnumeric()).all(axis = 1)
    
    

    df_good = df[index_good]
    len_out = df_good.shape[0]

    logger.info(f"Number of corrupt records {len_in - len_out}")

    return df_good





class SpellChecker:
    def __init__(self, threshold = 0.3):
        self.dict_pl = enchant.Dict("pl_PL")
        self.dict_en = enchant.Dict("en_EN")
        self.nlp_en = stanza.Pipeline("en")
        self.threshold = threshold
        
    def check(self, texts:list):
        """
        Core function
        """
        
        for i, text in enumerate(texts):
            texts[i] = [self.language_check(word) for word in text]
            
    
    def language_check(self, word:str):
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
            
            
    
    def best_match(self, text:str):
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
        
        if en and edit_distance(en[0], text)/len(text) < self.threshold:
            en = en[0]
            dist_en = edit_distance(en[0], text)
            
        if pl and edit_distance(pl[0], text)/len(text) < self.threshold:
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
        
        
def read_files(path, ndays, prefixes = ["gov","opp"], dtype = None, filter_fun = None):
    """
    
    Iterator reading data
    
    Parameters
    ----------
    path : str
        path to the folder containing the files.
    ndays : int
        number of days to be read.
    prefixes : list
        Prefixes used to separate classes. The default is ["gov","opp"].

    Returns
    -------
    None.

    """
    
    if not os.path.exists(path):
        raise ValueError("Incorrect path")
        
    files = os.listdir(path)
    if not files:
        raise ValueError("Empty path")
        
        
    files = sorted([fname for fname in files if "csv" in fname])
    
        
    
    for _ in range(ndays):
        data = pd.DataFrame() 
        for prefix in prefixes:
            ind = files.index([elem for elem in files if prefix in elem][0])
            tmp = pd.read_csv(os.path.join(path, files.pop(ind)), index_col = 0, dtype = dtype)
            tmp["source"] = prefix
            data = data.append(tmp)
        data.reset_index(inplace = True, drop = True)
        if filter_fun:
            data = filter_fun(data)
        yield data

    
def filter_data(path_source, path_target, nfiles, filter_fun, **kwargs):
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
    #clean:
    if os.path.isfile(path_target):
        os.remove(path_target)
        
    for data in tqdm(read_files(path_source, nfiles, prefixes = ["gov", "opp"], 
                                dtype = str, 
                                filter_fun = filter_fun, **kwargs)):
        if os.path.isfile(path_target):
            data.to_csv(path_target,  mode = "a", header = False)
        else:
            data.to_csv(path_target, mode = "w")
            
            
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
        
def tweets_datesplit(path_source:str, dir_target:str = None, chunksize:int = 100000):
    
    dir_source, file_source = os.path.split(path_source)
    
    if dir_target is None:
        dir_target = dir_source
        
    for df in tqdm(pd.read_csv(path_source, index_col = 0, 
                          dtype = str, iterator = True, error_bad_lines=False, chunksize=chunksize)):
        
        df = check_errors(df)
        
        date = pd.to_datetime(df['created_at']).dt.date
            
        
        for day, df in df.groupby(date):
            
            file_target = file_source.split('.')[0] + '_' + day.strftime('%Y_%m_%d') + '.csv'
            
            path_target = os.path.join(dir_target, file_target)
            
            if os.path.isfile(path_target):
                
                df.to_csv(path_target, mode = 'a', header = False)
            
            else:
                
                df.to_csv(path_target)
            
                
                
                
if __name__ == "__main__":
    path_gov = '/media/piotr/SAMSUNG/data/gov/gov_tweets.csv'
    tweets_datesplit(path_gov, chunksize = 100000)
    
        

    


        
            
            
    