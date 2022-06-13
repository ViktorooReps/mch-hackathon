from bank_embeddings.comparator import Comparator
from bank_embeddings.feature_extraction.bert import BertFeatureExtractor
from feature_extraction.text_pairs import *
from fact_extraction.entity_extractor import EntityExtractor
import pandas as pd
import nltk
import joblib
import os
import numpy as np

class Pipeline:
    def __init__(self, data_dir='bank_embeddings/input', top_k=10, use_title=False):
        self.sf = Comparator(data_dir)
        self.aofe = ArticleOriginFeatureExtractor(
            text_chunker=nltk.tokenize.sent_tokenize,
            entity_extractor=EntityExtractor(),
            feature_extractor=BertFeatureExtractor().extract_features)
        self.exp_path = 'lgbm_model/exp_4'
        self.model = joblib.load(os.path.join(self.exp_path, 'lgbm_model.pkl'))
        self.threshold = 0.5
        self.top_k = top_k
        self.use_title = use_title

    def __call__(self, art):
        titles, texts, similarities = self.sf.get_source(art, top_k=self.top_k, \
                                                        use_title=self.use_title)
        print('*' * 100)
        for title, text, similarity in zip(titles, texts, similarities):
            print(f"{title:>100} : {similarity:.4f}")
        print('*' * 100)
        po = texts
        res = self.aofe.extract_features(art, po)
        features = res.features
        features_sorted = dict(sorted(features.items(), key=lambda x: x[0]))
        arr = []
        for key in features_sorted:
            arr.append(features_sorted[key])
        features = np.array(arr)
        probabilities = self.model.predict_proba(features.reshape(1, -1))

        print(f'FAKE: {probabilities[0][1]}')
        return res, probabilities[0][1], self.threshold


def pipeline_factory():
    pipeline = Pipeline()

    return pipeline