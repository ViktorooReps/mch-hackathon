# Kadmus Fake Detection
Распознавание фейковых новостей.

Система способна определять новости, в которых произошло искажение фактов относительно информации в первоисточнике, а также выделять конкретные факты, которые, по мнению системы, были искажены.

Точность определения фейковости новостей: ...%


## Принцип работы:

* Форма на предложенной веб-странице принимает текст новости, для которой требуется узнать вероятность того, что новость фейковая.
* Текст предобрабатывается, после чего для этого текста в базе новостей ищется наиболее вероятный кандидат на первоисточник. Если все новости в базе сильно отличаются по тематике от запрошенной, то пользователю выводится сообщение о том, что первоисточник не найден
* В случае, если удалось найти достаточно правдоподобного кандидата на первоисточник, этот кандидат и запрошенный текст разбиваются по предложениям, после чего эти предложения сравниваются между собой - для каждого предложения из первоисточника ищется соответствующее предложение из запрошенного текста. Из предложений извлекаются факты и сравниваются между собой, и, на основе этого сравнения, алгоритм выдает вероятность того, что новость является фейковой. 


## Технические детали:

Предположения:
* Предполагаем, что новости с портала mos.ru являются достоверными. Предположение корректноработают профессионалы, которые а) максимально быстро публикуют информацию и б) максимально качественно занимаются факт-чеккингом. 


## Пайплайн 

### Обучение:

* Сохраняем банк новостей с mos ru, делим на трейн и тест
* Генерируем и сохраняем по трейну новый датасет - (исходный текст, парафразированный, семантизированный, парафразированный и семантизированный)
* Дополняем этот csv-датасет эмбеддингами каждого из 4 текстов 
* Формируем новую трейн-выборку: идем в цикле и добавляем в датасет 4 пары эмбеддингов на каждой итерации: (src, src), (src, par), (src, sem), (src, par_sem) 
* Обучаем модель:
    * Применяем модель генерации признаков по паре эмбеддингов
    * Подаем в классификатор LGBM вместе с меткой фейк/не фейк 

### Валидация:

* Формируем датасет по test.csv аналогично трейну и считаем метрику

### Тест в реальной жизни:

* Принимаем на вход текст статьи, которую надо проверить на фейковость
* Берем эмбеддинг Берта от всего текста
* Ищем ближайшего соседа по эмбеддингу из банка новостей  - говорим, что это кандидат на первоисточник
    * Если радиус больше порога - говорим что нет первоисточника, заканчиваем.
    * Если радиус меньше порога - идем дальше
* Применяем модель генерации признаков по паре (src, mb_fake) -> вектор признаков
* Применяем LGBM, который дает вероятность того что письмо фейк 


## Изменения текста 

1. Семантически нейтральные 
* Backtranslation
* Парафраз  
* Удалять предложения/абзацы, тогда полученную новость можно считать достоверной (меньше количество информации по сравнению с первоисточником не считается фейком)


2. Меняющие семантику 
* Распознавать именованные сущности и замена их на похожие 
	* С помощью предобученной модели из spacy для русского языка (данный подход позволяет распознавать сущности верно в 98-99% случаях). Заменять в тексте локации, организации и личности с определенной вероятностью. Замена подбирается случайно из списка сущностей того же типа, этот список сформирован из выявленных сущностей новостей с mos.ru
	* С помощью предобученной модели из spacy для английского языка: DATE, TIME, PERCENT, CARDINAL 
	* Применялась дополнительная фильтрация сущностей, проверки на корректность
* Удалять предложения/абзацы, тогда исходную новость можно считать фейковом (содержатся лишние факты, которых нет в первоисточнике)


В итоге для каждой исходной статьи из обучающей выборки были сгенерированы 3 измененных: перефразированная, с изменением сущностей, перефразированная с изменением сущность. Таким образом, исходная и перефразированная новости являются достоверными, а остальные две - фейковыми. 


### Именнованные сущности:

LOC - локация
ORG - организация
PER - личность 
PERCENT - процент
TIME - время
DATE - дата
CARDINAL - числа


## Описание модулей: 

semantic_modification.py - модуль для изменения фактов в тексте
paraphraser.py - модуль, составляющий парафразу для абзацев в тексте. 
datamodel.py - обертки для данных
augmentation_script.py - скрипт для получения аугментированного датасета для обучения классификатора

lgbm_model - модуль LGBM-классификатора 
feature_extraction - модуль для извлечения признаков из пары текстов - первоисточника и фейка
fact_extraction - ...
data - обкаченные краулером данные 
crawlers - скрипты для краулинга mos.ru
bank_embeddings - модуль для получения эмбеддингов из текстов

crawl_mos.sh - bash-скрипт для запуска краулера новостей с mos.ru
init.sh - модуль для настройки окружения

requirements.txt - зависимости проекта

web_app.py - код веб-приложения




