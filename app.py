from flask import Flask, request, jsonify, render_template
from demo_app import TextQA

app = Flask(__name__, template_folder='/templates')

@app.route('/')
def home():
    return render_template('index.html')

# 假设你已经有一个 TextQA 实例
text_qa = TextQA(txt_file_path='data/corpus.json')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')

    if question:
        text_qa.add_question(question)
        answer = text_qa.answer_questions()
        return jsonify({'answer': answer})
    else:
        return jsonify({'error': 'No question provided'}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8080)