from feature_extraction.bert import BertFeatureExtractor
from bank_embeddings.comparator import Comparator
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
class SourceFinder:
    def __init__(self, white_list: pd.core.frame.DataFrame, device='cpu'):
        self.device = 'cpu'
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.df = white_list
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        self.df.index = np.arange(len(self.df))
        self.fe = BertFeatureExtractor()
        self.features = self.make_features()
        self.article_numbers = self.make_art_nums()

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
        cmp = Comparator(0, 0, self.features)
        art_feature = np.array(self.fe.extract_features(text)).reshape(1, -1)
        indices, similarity = cmp.search_nearest_neightbours(art_feature, top_k=5, use_title=False)
        return (self.df.iloc[indices], similarity[indices])
