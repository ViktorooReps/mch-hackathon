import pickle
from argparse import ArgumentParser
from typing import Tuple, Optional

import nltk
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from tqdm import tqdm

from fact_extraction.entity_extractor import EntityExtractor
from feature_extraction.bert import BertFeatureExtractor
from feature_extraction.text_pairs import ArticleOriginFeatureExtractor


def create_features(from_file: str, *, start_idx: int = 0, end_idx: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    feature_extractor = ArticleOriginFeatureExtractor(
        text_chunker=nltk.tokenize.sent_tokenize,
        entity_extractor=EntityExtractor(),
        feature_extractor=BertFeatureExtractor().extract_features
    )

    csv_data = pd.read_csv(from_file)

    if end_idx is None:
        end_idx = len(csv_data.index)
    end_idx = min(len(csv_data.index), end_idx)

    total = end_idx - start_idx

    y = csv_data['label'].to_numpy()[start_idx:end_idx]
    y: NDArray
    y.reshape((total, 1))

    x: Optional[DataFrame] = None

    for row_idx, row in tqdm(csv_data[start_idx:end_idx].iterrows(), total=total):
        feats = feature_extractor.extract_features(row['fake_text'], [row['src_text']]).features
        # feats['embedding_distance'] = minkowski(row['src_emb'], row['fake_emb'], 2)

        if x is None:
            x = DataFrame(columns=sorted(set(feats.keys())))
        x = x.append(feats, ignore_index=True)

    return x.to_numpy(), y


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()
    features = create_features('train_paired_data.csv', start_idx=args.start_idx, end_idx=args.end_idx)
    with open(f'cached_{args.start_idx}_{args.end_idx}.pkl', 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
