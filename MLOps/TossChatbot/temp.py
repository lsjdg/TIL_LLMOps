from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from loaders.builder import *
from chromadb.config import Settings


class Retriever(Chroma):

    def __init__(
        self, collection_name, embedding_function, persist_directory, client_settings
    ):
        """
        collection_name: str
            name of collection to load

        returns: retriever from collection
        """
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
            #persist_directory="data/db",
            client_settings=Settings(persist_directory="data/db", is_persistent=True),
        )

        self.retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    def get_retriever(self):
        return self.retriever
