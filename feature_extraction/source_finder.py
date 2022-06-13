from pandas import DataFrame

from bank_embeddings.comparator import Comparator
import pandas as pd
import numpy as np
import torch


class SourceFinder:
    def __init__(self, white_list: DataFrame, device='cpu'):
        self.device = 'cpu'
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.df = white_list
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        self.df.index = np.arange(len(self.df))
        self.features = self.make_features()
        self.article_numbers = self.make_art_nums()
        self.com = Comparator(self.df, None, self.features)

    def get_white_list(self):
        return self.df

    def make_features(self):
        features = np.array(list(self.df['features'].apply(lambda x: list(x))))
        return features

    def get_features(self):
        return self.features

    def make_art_nums(self):
        return np.array(self.df.index)

    def get_art_nums(self):
        return self.article_numbers

    def find_source(self, text):
        return self.com.get_source(text, top_k=5, use_title=False)

