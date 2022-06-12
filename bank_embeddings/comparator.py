from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from os.path import join

import warnings
warnings.filterwarnings("ignore")
from feature_extraction.bert import FeatureExtractor

class Comparator():
    def __init__(self, dataframe: pd.DataFrame, title_embeddings: np.array, text_embeddings: np.array):
        self.source_meta = dataframe
        
        self.title_source_embeddings = title_embeddings
        self.text_source_embeddings = text_embeddings

        self.embedder = FeatureExtractor()

    def get_source(self, text: str, *, top_k: int = 1, use_title: bool = False):
        embedding = np.array(self.embedder.extract_features(text))
        index, similarity = self.search_nearest_neightbours(embedding, top_k, use_title)
        return (self.source_meta.loc[index, "title"], self.source_meta.loc[index, "text"], similarity[index])

    def search_nearest_neightbours(self, embedding: np.array, top_k: int, use_title: bool):
        if use_title:
            similarity = cosine_similarity([embedding], self.title_source_embeddings)
        else:
            similarity = cosine_similarity([embedding], self.text_source_embeddings)
        similarity = similarity[0]
        indices = np.argsort(similarity)
        indices = indices[-top_k:]
        return indices, similarity

if __name__ == "__main__":
    dataset = 'train+test'
    use_title = False
    top_k = 50
    data_dir = 'input'

    if dataset == 'test' or dataset == 'train':
        meta = pd.read_csv(join(data_dir, f'{dataset}.csv'))
        text_embeddings = np.load(join(data_dir, f'{dataset}_text_bank_embeddings.npy'))
        title_embeddings = np.load(join(data_dir, f'{dataset}_title_bank_embeddings.npy'))
    else:
        train_meta = pd.read_csv(join(data_dir, f'train.csv'))
        train_text_embeddings = np.load(join(data_dir, f'train_text_bank_embeddings.npy'))
        train_title_embeddings = np.load(join(data_dir, f'train_title_bank_embeddings.npy'))

        test_meta = pd.read_csv(join(data_dir, f'test.csv'))
        test_text_embeddings = np.load(join(data_dir, f'test_text_bank_embeddings.npy'))
        test_title_embeddings = np.load(join(data_dir, f'test_title_bank_embeddings.npy'))

        meta = pd.concat([train_meta, test_meta]).reset_index(drop=True)
        text_embeddings = np.concatenate([train_text_embeddings, test_text_embeddings], axis=0)
        title_embeddings = np.concatenate([train_title_embeddings, test_title_embeddings], axis=0)

    # print(len(meta), text_embeddings.shape, title_embeddings.shape)
    print(f"Meta-information length : {len(meta)}")
    print(f"Text embeddings shape : {text_embeddings.shape}")
    print(f"Title embeddings shape : {title_embeddings.shape}")
    
    fakes = pd.read_csv(join(data_dir, 'provided_fakes.csv'))
    comparator = Comparator(meta,  title_embeddings, text_embeddings)
    for test_index in range(len(fakes)):
        print('*' * 100)
        fake_title, fake_text = fakes['title'][test_index], fakes['text'][test_index]
        titles, texts, similarities = comparator.get_source(fake_text, top_k=top_k, use_title=use_title)
        print(f"Fake title : {fake_title}")
        for title, text, similarity in zip(titles, texts, similarities):
            print(f"{title:>100} : {similarity:.4f}")
    print('*' * 100)
    # print(f"Source title : Суперсервис на портале mos.ru поможет участникам программы реновации докупить квадратные метры")
    # titles, texts, similarities = comparator.get_source('Суперсервис на портале mos.ru поможет участникам программы реновации докупить квадратные метры', top_k=top_k, use_title=use_title)
    # for title, text, similarity in zip(titles, texts, similarities):
    #         print(f"{title:>100} : {similarity:.4f}")

    
