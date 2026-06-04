import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("Customer Service Chatbot 🤖")

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_db()

def get_answer(question):
    docs = db.similarity_search(question, k=1)
    if docs:
        return docs[0].page_content # Just return the answer text
    return "Sorry, I couldn't find an answer for that."

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask me a question about our services"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})