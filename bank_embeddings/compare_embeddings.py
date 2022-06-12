"""
Данный модуль создает файлик embedding_similarity.png
который вероятно можно будет показать оргам чтобы они посмотрели 
насколько похожи их измененные тексты между собой
"""

from sklearn.metrics.pairwise import cosine_distances
from feature_extraction.bert import FeatureExtractor

import pandas as pd
import numpy as np
import tqdm

import plotly.express as px
import os
from os.path import join

if __name__ == '__main__':
    data_dir = 'input'
    if True or not os.path.isfile(join(data_dir, 'distances_fakes.npy')):
        fe = FeatureExtractor(device='cuda:0')
        fakes = pd.read_csv(join(data_dir, 'provided_fakes.csv'))

        embeddings = []
        for fake in tqdm.tqdm(fakes.text):
            text_embedding = fe.extract_features(fake)  
            embeddings.append(text_embedding)
        embeddings = np.stack(embeddings, axis=0)
        embeddings_cpy = np.array(embeddings).copy()
        print(embeddings.shape, embeddings_cpy.shape)
        distances = np.array(cosine_distances(embeddings, embeddings_cpy))
    else:
        distances = np.load('distances_fakes.npy')
    fig = px.imshow(1 - distances, text_auto=True)
    fig.write_image(join(data_dir, 'embeddings_similarity.png'), scale=4)
    np.save(join(data_dir, 'distances_fakes.npy'), distances)