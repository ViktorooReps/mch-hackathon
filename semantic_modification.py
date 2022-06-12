import copy
import pickle
import spacy
import re 

from fact_extraction.helper import *

Config = {
    'LOC' : 0.5,
    'PER': 0.5,
    'ORG': 0.5,
    'TIME' : 0.5,
    'DATE' : 0.5,
    'CARDINAL': 0.5,
    'PERCENT': 0.5,
    'ALL' : 1.0
}

class SemChanger:
    def __init__(self, ru_space_path="ru_core_news_lg", 
                 en_space_path="en_core_web_lg",
                mapping_path='data/ner_mapping_clean.pkl',
                config=Config):
        self.ru_nlp = spacy.load(ru_space_path)
        self.en_nlp = spacy.load(en_space_path)
        
        
        with open(mapping_path, 'rb') as fout:
            self.mapping =  pickle.load(fout)
        
        self.config = config
    
    
    def change_condition(self, change_proba):
        return np.random.rand() < change_proba
    

    def ru_ner(self, input_text):

        changed = False

        ru_doc = self.ru_nlp(input_text)
        replaced_text = copy.deepcopy(input_text)

        starts = [0] + [ent.end_char for ent in ru_doc.ents]
        ends = [ent.start_char for ent in ru_doc.ents] + [len(input_text)]
        entities = [np.random.choice(list(self.mapping[ent.label_]), 1)[0] for ent in ru_doc.ents] + ['']
        src_texts = [ent.text for ent in ru_doc.ents] + ['']
        entities_types = [ent.label_ for ent in ru_doc.ents] + ['ALL']

        result = ""
        for start, end, entity, src_text, ent_type in zip(starts, ends, entities, src_texts, entities_types):
            result += input_text[start:end]
            if self.change_condition(self.config[ent_type]):
                changed = True
                result += entity
            else:
                result += src_text

        return result, changed
    
    
    def en_ner(self, input_text):
        changed = False

        en_doc = self.en_nlp(input_text)

        starts = [0]
        ends = []
        entities = []

        for ent in en_doc.ents:
            if ent.label_ == 'TIME':
                to_change = self.change_condition(self.config['TIME'])
                if to_change:
                    changed = True
                
                to_use, replaced = get_filtered_time(ent.text)
                replaced = np.random.choice(list(self.mapping[ent.label_]), 1)[0] if to_change else ent.text
                if to_use:
                    starts.append(ent.end_char)
                    ends.append(ent.start_char)
                    entities.append( replaced )

            if ent.label_ == 'DATE':
                to_change = self.change_condition(self.config['DATE'])
                if to_change:
                    changed = True
                    
                to_use, replaced = get_filtered_date(ent.text)
                replaced = np.random.choice(list(self.mapping[ent.label_]), 1)[0] if to_change else ent.text
                if to_use:
                    starts.append(ent.end_char)
                    ends.append(ent.start_char)
                    entities.append( replaced )

            if ent.label_ == 'CARDINAL':
                to_change = self.change_condition(self.config['CARDINAL'])
                if to_change:
                    changed = True
                
                to_use, replaced = get_filtered_cardinal(ent.text)
                replaced = replaced if to_change else ent.text
                if to_use:
                    starts.append(ent.end_char)
                    ends.append(ent.start_char)
                    entities.append( replaced )

        ends += [len(input_text)]
        entities += ['']

        result = ""
        for start, end, entity in zip(starts, ends, entities):
            result += input_text[start:end] + entity

        return result, changed
    
    
    def perc_ner(self, input_text):
        if self.change_condition(self.config['PERCENT']):
            return re.sub(r"\d+%", str(np.random.randint(0, 101, size=1)[0]) + "%", input_text), True
        return input_text, False
    
    
    def change_facts(self, input_text):
        while True:
            input_text, changed_1 = self.ru_ner(input_text)
            input_text, changed_2 = self.en_ner(input_text)
            input_text, changed_3 = self.perc_ner(input_text)
            if sum([changed_1, changed_2, changed_3]) > 0:
                break

        return input_text
