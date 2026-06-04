import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = []
csv_folder = "dataset.csv"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for file in os.listdir(csv_folder):
    file_path = os.path.join(csv_folder, file)
    
    # Handle CSV files
    if file.endswith(".csv"):
        print(f"Reading CSV: {file}")
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        for _, row in df.iterrows():
            answer = str(row['Answer']).strip()
            if answer and answer.lower() != 'nan':
                documents.append(Document(page_content=answer, metadata={"source": file}))
    
    # Handle TXT files - NEW
    elif file.endswith(".txt"):
        print(f"Reading TXT: {file}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": file}))

print(f"Loaded {len(documents)} documents from all sources")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)
db.save_local("vectorstore")
print("Knowledge base updated successfully.")