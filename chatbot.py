from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

def get_answer(question):

    docs = db.similarity_search(
        question,
        k=3
    )

    answer = ""

    for doc in docs:
        answer += doc.page_content + "\n\n"

    return answer