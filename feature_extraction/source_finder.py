from feature_extraction.bert import BertFeatureExtractor
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
class SourceFinder:
    def __init__(self, white_list: pd.core.frame.DataFrame, model_kwargs: dict, device='cpu'):
        self.device = 'cpu'
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.df = white_list
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        self.df.index = np.arange(len(self.df))
        self.cls_model = KNeighborsClassifier(**model_kwargs)
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
        self.cls_model.fit(self.features, self.article_numbers)
        art_feature = np.array(self.fe.extract_features(text)).reshape(1, -1)
        art_num = self.cls_model.predict(art_feature).reshape(1, -1)[0]
        source_feature = np.array(self.df.loc[art_num, 'features'].values[0]).reshape(1, -1)
        distance = pairwise_distances(art_feature, source_feature, metric=self.cls_model.metric)
        return (self.df.iloc[art_num], distance)
