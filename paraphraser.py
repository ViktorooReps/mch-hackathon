from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from fact_extraction.entity_extractor import EntityExtractor

class Paraphraser():
    def __init__(self, beams=3, grams=4, do_sample=False, paraphraser="cointegrated/rut5-base-paraphraser", device='cpu'):
        self.device = 'cpu'
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.beams = beams
        self.grams = grams
        self.do_sample = do_sample
        self.tokenizer = AutoTokenizer.from_pretrained(paraphraser)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(paraphraser)
        self.model.to(self.device)
        self.extractor = EntityExtractor()

        
    def __call__(self, text):
        splitted_text = text.split('\n\n')
        modified_text = []
        for paragraph in splitted_text:
            
            entities = self.extractor.get_entities(paragraph)
            if len(entities) == 0 or np.random.rand() < 0.2:
                x = self.tokenizer(paragraph, return_tensors='pt', padding=True).to(self.model.device)
                max_size = int(x.input_ids.shape[1] * 1.5 + 10)
                out = self.model.generate(**x, encoder_no_repeat_ngram_size=self.grams,
                                          num_beams=self.beams,
                                          max_length=max_size,
                                          do_sample=self.do_sample)

                modified_text.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
            
            else:
                modified_text.append(paragraph)
            
        modified_text = "\n\n".join(modified_text)
        return modified_text

if __name__ == '__main__':
    prph = Paraphraser()
    text = 'Для пользователей портала «Узнай Москву» создали облегченную версию мобильного приложения. Ее могут скачать горожане, чьи смартфоны или планшеты обладают невысокой мощностью. Речь идет об устройствах с операционной системой Android, не поддерживающих технологию ARCore, которая необходима для использования функций дополненной реальности. Более легкая версия приложения специально адаптирована для работы на таких гаджетах.\n\nРанее владельцы моделей смартфонов прошлых поколений могли сталкиваться с трудностями при загрузке контента, например длительным ожиданием отображения на экране фотографий, а также информации о зданиях, памятниках или музеях.\n\n«Облегченный аналог онлайн-гида “Узнай Москву” будет полезен обладателям смартфонов, планшетов с невысокими техническими характеристиками, на которых установлена операционная система Android. Работать с полноценной версией приложения таким моделям зачастую сложно, контент долго загружается на смартфоны. Новая версия стала меньше весить, поэтому страницы приложения загружаются быстрее. При этом пользователям доступны практически все функции стандартной версии», — рассказали в пресс-службе Департамента информационных технологий.\n\nВ обновленном мобильном приложении портала «Узнай Москву» можно составлять познавательные маршруты по городу, знакомиться с историей зданий, улиц и районов, узнавать интересные факты о достопримечательностях, расположенных поблизости, проходить увлекательные онлайн-квесты, изучать биографии известных личностей: архитекторов и художников, ученых и литераторов, крупных военачальников, купцов, живших в Москве.\n\nПодобные возможности есть и у тех, у кого установлен классический аналог приложения. Отличие облегченной версии лишь в том, что ее пользователи не могут рассматривать на смартфоне или планшете столичные объекты в режиме дополненной реальности.\n\nСкачать облегченное приложение на смартфон или планшет можно в Google Play.\n\nПортал «Узнай Москву» — это совместный проект столичных департаментов информационных технологий, культуры, культурного наследия, образования и науки. Интерактивный городской гид содержит фотографии и описания более 2,2 тысячи зданий, 670 памятников, 350 музеев, а также сведения о 334 исторических личностях. На портале опубликовано свыше 210 маршрутов по разным районам города. Проект доступен в формате мобильного приложения.'
    print('0:', text)
    print('1:', prph(text))
