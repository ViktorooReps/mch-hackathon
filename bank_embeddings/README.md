### Преобразование raw-текстов из ТЗ в .csv
python3 create_valid.py

### Посмотреть на матрицу расстояний для текстов из ТЗ
python3 compare_embeddings.py

### Построить эмбеддинги для текста + заголовков для теста + трейна
python3 compute_bank_embeddings.py

### Сделать query к банку
python3 comparator.py

Интерфейс компаратора следующий:

Создается класс ``` comparator = Comparator(meta,  title_embeddings, text_embeddings) ``` в который подается мета информация о текстах из краулера, эмбеддинги заголовков и текстов.

Для того чтобы совершить запрос:

``` comparator_instance.get_source(text, top_k=top_k, use_title=use_title) ```
* флаг ```use_title``` отвечает за то, будет ли для матчинга использоваться заголовок или текст из банка
* флаг ```top_k``` настраивает число семплов из банка, которые вернет компаратор

### Что возвращает компаратор

```top_k``` троек (title, text, similarity) из банка.
