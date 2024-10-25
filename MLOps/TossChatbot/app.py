from flask import Flask, request, jsonify
from bots.chatbot import Chatbot
from loaders.builder import VectorStore
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

vectordb = vectordb = VectorStore(
    collection_name="toss_kids_account",
    embedding_function=OpenAIEmbeddings(),
    client_settings=Settings(persist_directory="data/db", is_persistent=True),
)
llm = Chatbot(
    model_name="gpt-4o-mini", streaming=False, temperatrue=0, vectorstore=vectordb
)

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("input")
    response = llm.ask(query)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
