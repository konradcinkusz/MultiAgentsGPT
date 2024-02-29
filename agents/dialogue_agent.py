from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
        self.persist_directory = "empty"
        self.vectordb : Chroma

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def _apply_vector_store_to_message_history(self):
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        if self.persist_directory != "empty":
            # TODO : infer the question from message_history, because I want to reflect on current documents
            question = "is there an email i can ask for help"
            docs = self.vectordb.similarity_search(question, k=3) # k=3 numbers of documents that we wanna return
            self.vectordb.persist()
            # Append the content of the first document as the last message in message_history
            self.message_history.append(docs[0].page_content)

    def send(self) -> str:
        self._apply_vector_store_to_message_history()
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content
    
    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

    def create_vector_store(self, loaders: List[PyPDFLoader]):
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(docs)

        embedding = OpenAIEmbeddings()
        
        self.vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding, # Assuming 'embedding' is an attribute of DialogueAgent
            persist_directory=self.persist_directory
        )

        self.vectordb.persist()