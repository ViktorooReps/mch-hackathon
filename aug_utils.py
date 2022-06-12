import numpy as np
import re


months = [
"май", "мая", "маю", "маем", "мае",
"июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр", "январ", "феврал", "март", "апрел",
]


time_words = ['до', 'и', 'начнется']


def get_filtered_date(inputs):
    inputs = inputs.replace('-', ' - ')
    inputs = inputs.replace('\n', ' ')
    if '№' in inputs:
        return False, None
    tokens = inputs.lower().split()
    min_len = 1
    res = []
    for token in tokens:
        if token == '-':
            min_len += 1
        if sum([letter.isalpha() for letter in token]):
            if check_if_month(token):
                res.append(token)
        else:
            res.append(token)

    if len(res) < min_len:
        return False, None
    
    return True, ' '.join(res).strip('-').strip()


def get_filtered_time(inputs):
    inputs = inputs.replace('-', ' ')
    inputs = inputs.replace('\n', ' ')
    
    words = inputs.split()
    res = []
    for word in words:
        if word.isalpha():
            if word not in time_words:
                return False, None
            res.append(word)
        else:
            for letter in word:
                if not (letter in ',:. ' or letter.isdigit()): 
                    return False, None
            res.append(word)
        
    return True, ' '.join(res).strip('-').strip()


def get_filtered_cardinal(inputs):
    inputs = inputs.replace('\n', ' ').strip()
    for letter in inputs:
        if not (letter in ',.- ' or letter.isdigit()):
            return False, None
    
    return True, str(np.random.randint(0, 100_000, size=1)[0])