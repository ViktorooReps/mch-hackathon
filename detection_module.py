from feature_extraction.source_finder import *
from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.text_pairs import *
from fact_extraction.entity_extractor import EntityExtractor
import pandas as pd
import nltk
import joblib
import os

class Pipeline:
    def __init__(self):
        self.white_list = pd.read_pickle('white_list.pkl')
        self.sf = SourceFinder(white_list=self.white_list)
        self.aofe = ArticleOriginFeatureExtractor(
            text_chunker=nltk.tokenize.sent_tokenize,
            entity_extractor=EntityExtractor(),
            feature_extractor=BertFeatureExtractor().extract_features)
        self.exp_path = 'lgbm_model/exp_4'
        self.model = joblib.load(os.path.join(self.exp_path, 'lgbm_model.pkl'))
        self.threshold = 0.5

    def __call__(self, art):
        article_info, distance = self.sf.find_source(art)
        po = article_info['text'].values
        res = self.aofe.extract_features(art, po)
        features = res.features
        features_sorted = dict(sorted(features.items(), key=lambda x: x[0]))
        arr = []
        for key in features_sorted:
            arr.append(features_sorted[key])
        features = np.array(arr)
        probabilities = self.model.predict_proba(features.reshape(1, -1))
        return res, probabilities[0][1], self.threshold


def pipeline_factory():
    pipeline = Pipeline()

    return pipeline

