import os
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings


db_path = 'kaggle/vector_db'
file_path = 'kaggle/600K_USHousingProperties.csv'
os.environ["OPENAI_API_KEY"] = 'sk-proj-YsE5dZkWvxqa3dJpGRhZT3BlbkFJ2zQSyvln9eq6cKEf2dau'

# ------------ Load data --------------
loader = CSVLoader(file_path=file_path)
dataset = loader.load()
dataset = dataset[:10]

# ----------- Embed and store database ----------------
db = Chroma.from_documents(dataset, OpenAIEmbeddings(), persist_directory=db_path)
db.persist()
print(f"Saved {len(dataset)} chunks to {db_path}.")

