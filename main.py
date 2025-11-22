import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Formaculture Chatbot", layout="centered")
st.title("🤖 Formaculture RAG Chatbot")
st.caption("Fast • Local • Llama 3.2 • Chat History Enabled")

# -----------------------------
# Load FAISS index
# -----------------------------
@st.cache_resource
def load_index():
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest")  # fast embedding model
    return FAISS.load_local("faiss_index/", embeddings, allow_dangerous_deserialization=True)

db = load_index()

# -----------------------------
# Initialize Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display old chat messages
for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")


# -----------------------------
# RAG + Conversation Memory
# -----------------------------
def generate_answer(user_query):
    # Retrieve TOP-2 chunks from vectorstore
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(user_query)

    context = "\n\n".join([d.page_content for d in docs])

    # Build conversation history
    history = ""
    for role, text in st.session_state["messages"]:
        history += f"{role.upper()}: {text}\n"

    # Prompt with context + history
    prompt = f"""
You are a helpful AI assistant built for the Formaculture Internship Task.

You MUST use the context below + the conversation history to answer.

### CONTEXT ###
{context}

### CHAT HISTORY ###
{history}

### USER QUESTION ###
{user_query}

If answer is not in context, say: "I cannot find this in the documents."

AI:
"""

    llm = OllamaLLM(model="llama3.2")  # faster model
    answer = llm.invoke(prompt)
    return answer


# -----------------------------
# User Input
# -----------------------------
user_input = st.text_input("Ask something:")

if st.button("Send"):
    if user_input.strip():
        # Add to history
        st.session_state["messages"].append(("user", user_input))

        # Generate response
        with st.spinner("Thinking..."):
            response = generate_answer(user_input)

        st.session_state["messages"].append(("ai", response))

        # Rerun app to refresh chat
        st.rerun()
