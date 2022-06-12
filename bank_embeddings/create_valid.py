"""
Модуль создает файл provided_fakes.csv в котором содержатся 
тексты и заголовки статей из ТЗ, использются файлы
input/text_1.txt, input/title_1.txt,
input/text_2.txt, input/title_2.txt,
...
"""

import pandas as pd
from os.path import join

if __name__ == '__main__':
    num_papers = 8
    data_dir = 'input'
    
    id = range(num_papers)
    texts = []
    titles = []
    for id in range(1, num_papers + 1):
        with open(join(data_dir, f"text_{id}.txt"), 'r') as file:
            text = file.readlines()
        with open(join(data_dir, f"title_{id}.txt"), "r") as file:
            title = file.readlines()
        texts.append(text)
        titles.append(title)

    df = pd.DataFrame({
        'id' : id,
        'text' : texts,
        'title' : titles
    })
    df.to_csv(join(data_dir, 'provided_fakes.csv'), index=False)