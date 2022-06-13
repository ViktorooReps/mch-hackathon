import pickle
from typing import Tuple

import nltk
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from fact_extraction.entity_extractor import EntityExtractor
from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.text_pairs import ArticleOriginFeatureExtractor


def create_features(from_file: str) -> Tuple[NDArray, NDArray]:
    feature_extractor = ArticleOriginFeatureExtractor(
        text_chunker=nltk.tokenize.sent_tokenize,
        entity_extractor=EntityExtractor(),
        feature_extractor=BertFeatureExtractor().extract_features
    )

    csv_data = pd.read_csv(from_file)
    for row_idx, row in tqdm(csv_data.iterrows(), total=len(csv_data.index)):
        true_example_features = feature_extractor.extract_features(row['par_text'], [row['orig_text']]).features
        false_example_features_sem = feature_extractor.extract_features(row['sem_text'], [row['orig_text']]).features
        false_example_features_sem_par = feature_extractor.extract_features(row['sem_text'], [row['orig_text']]).features


if __name__ == '__main__':
    features = create_features('augmented_data.csv')
    with open('cached.pkl', 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
