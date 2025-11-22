# 🤖 Formaculture RAG Chatbot

A fast, local Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Llama 3.2. This chatbot allows you to query PDF documents using semantic search and get contextually relevant answers with conversation history support.


## ✨ Features

- **📚 PDF Document Processing**: Automatically loads and processes PDF documents from a corpus
- **🔍 Semantic Search**: Uses FAISS vector store for efficient similarity search
- **💬 Chat History**: Maintains conversation context across multiple queries
- **⚡ Fast & Local**: Powered by Llama 3.2 and EmbeddingGemma models via Ollama
- **🎨 Clean UI**: Simple and intuitive Streamlit interface

## 🏗️ Architecture

The project consists of two main components:

1. **Index Builder** (`build_index.py`): Processes PDFs and creates a FAISS vector index
2. **Chatbot Interface** (`main.py`): Streamlit app for querying the indexed documents

### How It Works

1. PDFs are loaded from the `Corpus/` directory
2. Documents are split into chunks (800 characters with 150 character overlap)
3. Chunks are embedded using EmbeddingGemma and stored in a FAISS index
4. User queries retrieve the top-2 most relevant chunks
5. Llama 3.2 generates answers based on retrieved context and chat history

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- Required Ollama models:
  ```bash
  ollama pull llama3.2
  ollama pull embeddinggemma:latest
  ```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd formaculture-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your document corpus**
   - Create a `Corpus/` directory in the project root
   - Add your PDF documents to the `Corpus/` folder

4. **Build the FAISS index**
   ```bash
   python build_index.py
   ```
   This will create a `faiss_index/` directory with your vector embeddings.

## 💻 Usage

### Running Locally

Start the Streamlit app:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. Type your question in the text input field
2. Click "Send" to get an answer
3. The chatbot will:
   - Search for relevant context in your documents
   - Consider previous conversation history
   - Generate an informed response
4. If the answer isn't in the documents, it will let you know

## 📁 Project Structure

```
formaculture-chatbot/
├── build_index.py        # Script to create FAISS index from PDFs
├── main.py              # Streamlit chatbot interface
├── requirements.txt     # Python dependencies
├── Corpus/             # Directory for PDF documents (not included)
└── faiss_index/        # Generated FAISS vector store (not included)
```

## 🛠️ Configuration

### Adjusting Chunk Size

In `build_index.py`, modify the text splitter parameters:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Characters per chunk
    chunk_overlap=150    # Overlap between chunks
)
```

### Changing Retrieval Count

In `main.py`, adjust the number of retrieved documents:
```python
retriever = db.as_retriever(search_kwargs={"k": 2})  # Change k value
```

### Switching Models

Replace the model names in both files:
```python
# For embeddings
embeddings = OllamaEmbeddings(model="your-embedding-model")

# For LLM
llm = OllamaLLM(model="your-llm-model")
```

## 📦 Dependencies

- **streamlit**: Web interface framework
- **langchain**: LLM application framework
- **langchain-community**: Community integrations
- **langchain-ollama**: Ollama integration for LangChain
- **faiss-cpu**: Vector similarity search
- **pypdf**: PDF document loading
- **sentence-transformers**: Embedding models support

## 🔒 Security Note

The app uses `allow_dangerous_deserialization=True` when loading the FAISS index. This is necessary for FAISS functionality but should only be used with trusted index files.


---

**Note**: Make sure Ollama is running before starting the application, as the chatbot depends on local Ollama models for embeddings and text generation.
