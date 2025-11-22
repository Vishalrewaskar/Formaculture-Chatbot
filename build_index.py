import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def build_index():
    docs = []
    corpus_dir = "Corpus"

    for file in os.listdir(corpus_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(corpus_dir, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="embeddinggemma:latest") 
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")
    print("Index saved!")

if __name__ == "__main__":
    build_index()
