# Демонстрационная реализация RAG для сервиса [VseGPT.ru](https://vsegpt.ru/)

RAG - набор методов, направленных на то, чтобы 
- а) выбирать из больших серий документов куски текста, соответствующих запросу пользователя
- б) добавлять эти релевантные куски в запрос к нейросети - чтобы нейросеть отвечала, исходя из реальной информации из документов.

Применяется в случае, когда у вас есть МНОГО документов, не влезающих в контекст и когда надо по ним отвечать на запросы пользователя.

Базируется на базе данных FAISS - https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/

Использование:
- Установите зависимости через requirements.txt
- Добавьте свой ключ VseGPT в нужные места кода в файлы rag_db_create.py и rag_db_search.py 
```python
os.environ["OPENAI_API_KEY"] = "your_vsegpt_key"
```
- Запустите файл rag_db_create. Он сделает следующее:
  - Разобьет файл sun.txt (информация о Солнце из Википедии) на кусочки (chunks)
  - Для каждого кусочка сделает embedding (векторное выражение смысла) через API
  - Получившуюся базу сохранит в docs_db_index (т.к. базу имеет смысл сделать 1 раз)
- Запустите файл rag_db_search. Он сделает следующее:
  - Загрузит базу из docs_db_index
  - Найдет 3 чанка из базы данных, самых похожих на запрос пользователя
  - Добавит эти чанки в запрос к ChatGPT как точную информацию
  - Получит результат.

В текущем варианте:
- Вопрос: "Расстояние от Земли до Солнца?"
- Будет получен ответ: "Расстояние от Земли до Солнца составляет приблизительно 149,6 миллионов километров, что примерно равно астрономической единице."