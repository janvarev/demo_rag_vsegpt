import os

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

# place your VseGPT key here
os.environ["OPENAI_API_KEY"] = "your_vsegpt_key"

def create_search_db(file_text,
                        knowledge_base_link,
                        chunk_size=1024,
                        chunk_overlap=200):

    splitter = RecursiveCharacterTextSplitter(['\n\n', '\n', ' '], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_chunks = []

    # splitting to chunks
    for chunkID,chunk in enumerate(splitter.split_text(file_text)):
        source_chunks.append(Document(page_content=chunk, \
                            metadata={'source': knowledge_base_link,
                                      'chunkID': chunkID}))

    if len(source_chunks) > 0:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_base = "https://api.vsegpt.ru/v1/")
        db = FAISS.from_documents(source_chunks, embedding_model)
        db.save_local("docs_db_index")
        print("Docs.db search index created!")

    # return db

if __name__ == "__main__":
    # Read text from file
    with open("sun.txt", "r", encoding="utf-8") as file:
        file_text = file.read()

    # Link to the knowledge base, can be a URL or some identifier string
    knowledge_base_link = "sun_knowledge_base"

    # Run the create_search_db function
    create_search_db(file_text, knowledge_base_link)
