from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from loaders.builder import *


class Chatbot:

    def __init__(self, model_name, streaming, temperatrue, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperatrue
        )
        self.template = """당신은 Toss Bank에서 만든 금융 상품을 설명해주는 챗봇입니다.
        주어진 검색 결과를 바탕으로 답변하세요.
        검색 결과에 없는 내용이라면, 답변할 수 없다고 하세요.
        존댓말로 정중하게 대답해주세요.
        {context}

        Question: {question}
        Answer:
        """
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def ask(self, query):

        prompt = PromptTemplate.from_template(self.template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type_kwargs={"prompt": prompt},
            retriever=self.retriever,
            return_source_documents=False,
        )

        response = qa_chain.invoke(query)
        answer = response["result"]

        return answer

    def get_template(self):
        return self.template


if __name__ == "__main__":
    load_dotenv()
    chatbot = Chatbot(
        model_name="gpt-4o-mini",
        streaming=False,
        temperatrue=0,
    )

    print(chatbot.get_template())
