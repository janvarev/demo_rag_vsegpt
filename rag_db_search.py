import os

from openai import OpenAI

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# place your VseGPT key here
os.environ["OPENAI_API_KEY"] = "your_vsegpt_key"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_base = "https://api.vsegpt.ru/v1/")

def run_gpt_query(system, user_query, search_db):

    docs = search_db.similarity_search(user_query, 3) # число результатов схожих по эмбеддингу

    message_content = '\n\n'.join([doc.page_content for i, doc in enumerate(docs)])
    print('Используем следующую найденную информацию: ',message_content)
    print("-----")

    messages = [
      {"role": "system", "content": system},
      {"role": "user", "content": f'Ответь на вопрос пользователя, используя информацию из документа ниже.\n Документ: <doc>{message_content}</doc>. Не упоминай документ с информацией, указанный выше.\n Запрос пользователя: {user_query}'}]

    client = OpenAI(
        base_url="https://api.vsegpt.ru/v1",
    )

    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0.1)
    answer = completion.choices[0].message.content
    return answer

if __name__ == "__main__":
    db = FAISS.load_local("docs_db_index", embedding_model)
    answer = run_gpt_query("Ты - помощник, помогающий отвечать на вопросы","Расстояние от земли до солнца?",db)
    print("Финальный ответ:", answer)

