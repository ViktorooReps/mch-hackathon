import spacy
import re 

from aug_utils import *


class EntityExtractor:
    
    def __init__(self, ru_space_path="ru_core_news_lg", 
                 en_space_path="en_core_web_lg"):
        
        self.ru_nlp = spacy.load(ru_space_path)
        self.en_nlp = spacy.load(en_space_path)
    
    
    def ru_ner(self, input_text):

        ru_doc = self.ru_nlp(input_text)

        return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in ru_doc.ents]


    def en_ner(self, input_text):

        en_doc = self.en_nlp(input_text)

        result = []
        for ent in en_doc.ents:
            if ent.label_ == 'TIME':
                to_use, replaced = get_filtered_time(ent.text)
                if to_use:
                    result.append( (ent.text, ent.start_char, ent.end_char, ent.label_) )

            if ent.label_ == 'DATE':
                to_use, replaced = get_filtered_date(ent.text)
                if to_use:
                    result.append( (ent.text, ent.start_char, ent.end_char, ent.label_) )

            if ent.label_ == 'CARDINAL':
                to_use, replaced = get_filtered_cardinal(ent.text)
                if to_use:
                    result.append( (ent.text, ent.start_char, ent.end_char, ent.label_) )

        return result


    def perc_ner(self, input_text):
        result = []
        indices = [[m.start(), m.end()] for m in re.finditer(r"\d+%" , input_text)]
        for start, end in indices:
            result.append(text[start:end], start, end, 'PERCENT')
        return result


    def get_entities(self, input_text):
        results = []
        results += self.ru_ner(input_text)
        results += self.en_ner(input_text)
        results += self.perc_ner(input_text)
        return sorted(results, key=lambda w : w[1])