from sklearn.metrics.pairwise import cosine_distances
from feature_extraction.bert import FeatureExtractor

import pandas as pd
import numpy as np
import tqdm

import os
from os.path import join

if __name__ == '__main__':
    fe = FeatureExtractor()
    data_dir = 'input'
    for part in ['text', 'title']:
        for dataset in ['test', 'train']:
            if os.path.isfile(join(data_dir, f'{dataset}_{part}_bank_embeddings.npy')):
                continue
            df = pd.read_csv(join(data_dir, f"{dataset}.csv"))

            embeddings = []
            for num, fake in enumerate(tqdm.tqdm(df[part])):
                try:
                    text_embedding = fe.extract_features(fake)  
                except:
                    text_embedding = np.zeros(([1, 768]))
                embeddings.append(text_embedding)
            embeddings = np.stack(embeddings, axis=0)
            print(embeddings.shape)
            np.save(join(data_dir, f'{dataset}_{part}_bank_embeddings.npy'), embeddings)