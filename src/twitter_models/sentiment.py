from nltk import FreqDist
from itertools import chain
import numpy as np


class SentimentAnalyzer:
    def __init__(self, sent_dict: dict, a: int = 0.001, weighting='linear'):
        assert weighting in ['linear','inverse_frequency']
        self.sent_dict = sent_dict
        self.a = a
        self.weighting = weighting

    def fit(self, texts: list):
        dist = FreqDist(chain.from_iterable(texts))
        if self.weighting == 'inverse_frequency':
            total = sum(dist.values())
            self.weights = {k: self.a / (v / total) for k, v in dist.items()}
        elif self.weighting == 'linear':
            self.weights = dict.fromkeys(dist.keys(), 1)
        return self

    def predict(self, texts: list):
        sentiments = []
        for text in texts:
            sentiment = [self.sent_dict[w] for w in text if w in self.sent_dict]
            weights = [self.weights[w] for w in text if w in self.weights and w in self.sent_dict]
            if sentiment and weights:
                sentiments.append(np.average(np.array(sentiment), weights=np.array(weights)))
            else:
                sentiments.append(0)
        return sentiments


if __name__ == '__main__':
    import pandas as pd
    import ast
    from src.twitter_tools.preprocessing import Lemmatizer

    test = pd.read_csv('../../data/clean/gov_tweets_2020_02_23.csv')
    texts = test['lemmas'].apply(ast.literal_eval).tolist()
    sent_dict = pd.read_csv('/home/piotr/nlp/slownikWydzwieku01.csv', sep='\t', header=None)
    sent_dict.columns = ['word', 'unk', 'bin', 'tri', 'sentiment', 'so_pmi']

    # run lemmatization
    """
    lemmatizer = Lemmatizer()
    lemmas = lemmatizer.lemmatize(sent_dict['word'].tolist())[0]
    lemmas = [lemma[0][0] for lemma in lemmas]
    sent_dict['word'] = lemmas
    """



    # get sentiment scores
    sent_dict['score'] = 0
    sent_dict.loc[(sent_dict['sentiment'] > 0) & (sent_dict['so_pmi'] > 0), 'score'] = 1
    sent_dict.loc[(sent_dict['sentiment'] < 0) & (sent_dict['so_pmi'] < 0), 'score'] = -1
    sent_dict = sent_dict[['word', 'score']]
    sent_dict.drop_duplicates(subset=['word'], inplace=True)

    # get counts
    print(pd.value_counts(sent_dict['score']))
    print(sent_dict[sent_dict['score'] == 1]['word'].tolist())
    print(sent_dict[sent_dict['score'] == -1]['word'].tolist())

    # prepare dictionary
    sent_dict = sent_dict.set_index('word')['score'].to_dict()




    model = SentimentAnalyzer(sent_dict=sent_dict, weighting='linear')
    model.fit(texts)
    preds = model.predict(texts)
    test['sentiment'] = preds
    for text in test.sort_values('sentiment', ascending=True)['full_text'].iloc[:20]:
        print(text)

    print(np.quantile(test['sentiment'], np.linspace(0, 1, 10)))
    print(np.mean(test['sentiment']))
    print(np.median(test['sentiment']))
    print(sum(test['sentiment'] < 0)/test.shape[0])
    print(sum(test['sentiment'] == 0)/test.shape[0])
