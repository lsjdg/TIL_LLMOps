from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from loaders.builder import *
from chromadb.config import Settings


class Chatbot(ChatOpenAI):

    def __init__(self, model_name, streaming, temperatrue, retriever):
        """
        initialize chatbot
        """
        super(ChatOpenAI, self).__init__(
            model_name=model_name, streaming=streaming, temperature=temperatrue
        )
        load_dotenv()

        self.retriever = retriever

        self.template = """당신은 Toss Bank에서 만든 금융 상품을 설명해주는 챗봇입니다.
            주어진 검색 결과를 바탕으로 답변하세요.
            검색 결과에 없는 내용이라면, 답변할 수 없다고 하세요.
            존댓말로 정중하게 대답해주세요.
            {context}

            Question: {question}
            Answer:
        """

    def ask(self, query):

        prompt = PromptTemplate.from_template(self.template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self,
            chain_type_kwargs={"prompt": prompt},
            retriever=self.retriever,
            return_source_documents=False,
        )

        response = qa_chain.invoke(query)
        answer = response["result"]

        return answer


class Retriever:

    def __init__(self, collection_name):
        """
        collection_name: str
            name of collection to load

        returns: retriever from collection
        """
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
            persist_directory="data/db",
            client_settings=Settings(persist_directory="data/db", is_persistent=True),
        )

        self.retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    def get_retriever(self):
        return self.retriever
