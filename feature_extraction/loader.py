import pickle
from typing import Tuple, Optional

import nltk
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.spatial.distance import minkowski
from tqdm import tqdm

from fact_extraction.entity_extractor import EntityExtractor
from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.text_pairs import ArticleOriginFeatureExtractor


def create_features(from_file: str, *, first_n: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    feature_extractor = ArticleOriginFeatureExtractor(
        text_chunker=nltk.tokenize.sent_tokenize,
        entity_extractor=EntityExtractor(),
        feature_extractor=BertFeatureExtractor().extract_features
    )

    csv_data = pd.read_csv(from_file)

    if first_n is None:
        first_n = len(csv_data.index)
    first_n = min(len(csv_data.index), first_n)

    y = csv_data['label'].to_numpy()[:first_n]
    y: NDArray
    y.reshape((first_n, 1))

    x: Optional[DataFrame] = None

    for row_idx, row in tqdm(csv_data[:first_n].iterrows(), total=first_n):
        feats = feature_extractor.extract_features(row['fake_text'], [row['src_text']]).features
        # feats['embedding_distance'] = minkowski(row['src_emb'], row['fake_emb'], 2)

        if x is None:
            x = DataFrame(columns=set(feats.keys()))
        x = x.append(feats, ignore_index=True)

    return x.to_numpy(), y


if __name__ == '__main__':
    features = create_features('train_paired_data.csv')
    with open('cached.pkl', 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
