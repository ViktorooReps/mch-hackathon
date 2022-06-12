# !pip install thefuzz
# !pip3 install https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.3.0/ru_core_news_lg-3.3.0.tar.gz
# !pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0.tar.gz
# !pip install transformers
# !pip install sentencepiece
from argparse import ArgumentParser

from paraphraser import Paraphraser
from semantic_modification import SemChanger
import pandas as pd
from tqdm import tqdm
from multiprocessing.dummy import Pool, Queue
import shelve
import pickle


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_file', type=str, default='data/mos/validation.csv')
    args = parser.parse_args()

    source_file = args.source_file
    source_file_name = source_file.split('.')[0]
    df = pd.read_csv(source_file)
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    df.head(1)

    prph = Paraphraser(device='cuda')
    changer = SemChanger()

    PROCESSES = 1
    queue = Queue()

    for link in range(df.shape[0]):
        queue.put(link)

    queue.empty()

    rows = {i: [] for i in range(PROCESSES)}

    def process_page_wrapper(i):

        while not queue.empty():
            with shelve.open(f'backup_{i}_{source_file_name}.db') as db:
                num = queue.get()
                row = df.iloc[num]

                text = row['text']
                par_text = prph(text)
                sem_text = changer.change_facts(text)
                par_sem_text = changer.change_facts(par_text)
                new_row = {'art_num': num, 'orig_text': text, 'par_text': par_text,
                               'sem_text': sem_text, 'par_sem_text': par_sem_text}
                rows[i].append(new_row)
                db[str(num)] = new_row

                with lock:
                    pbar.update(1)


    with Pool(processes=PROCESSES) as pool, tqdm(total=queue.qsize()) as pbar:
        lock = pbar.get_lock()
        pool.map(process_page_wrapper, range(pool._processes))

    pool.join()

    with open(f'backup_result_{source_file_name}', 'wb') as fout:
        pickle.dump(rows, fout)
