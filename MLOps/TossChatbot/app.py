from flask import Flask, request, jsonify
from chatbot.chatbot import Chatbot, Retriever


llm = Chatbot(
    model_name="gpt-4o-mini",
    streaming=False,
    temperatrue=0,
    retriever=Retriever("tossbank_kids_account").get_retriever(),
)

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("input")
    response = llm.ask(query)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
