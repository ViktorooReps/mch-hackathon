from paraphraser import Paraphraser
from semantic_modification import SemChanger
import pandas as pd
from tqdm import tqdm
from multiprocessing.dummy import Pool, Queue
import shelve
import pickle


df = pd.read_csv('data/mos/train.csv')
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
df.head(1)


prph = Paraphraser(device='cuda')
changer = SemChanger()


PROCESSES = 4
queue = Queue()

for link in range(df.shape[0]):
    queue.put(link)
    
queue.empty()


rows = {i : [] for i in range(PROCESSES)}
def process_page_wrapper(i):
    
    while not queue.empty():
        with shelve.open(f'backup_{i}.db') as db:
            num = queue.get()
            num = df.shape[0] - num - 1
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


with open('backup_res', 'wb') as fout:
    pickle.dump(rows, fout)