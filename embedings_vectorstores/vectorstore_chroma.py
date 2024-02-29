import os
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# https://python.langchain.com/docs/modules/data_connection/text_embedding/
# think about text as a vector space
embedding = OpenAIEmbeddings()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(base_dir, 'assets')

loaders = [
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture01.pdf")),
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture01.pdf")),
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture02.pdf")),
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture03.pdf"))
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)

persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3) #k=3 numbers of documents that we wanna return
print(len(docs))
print(docs[0].page_content)
vectordb.persist()