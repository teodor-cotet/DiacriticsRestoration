from flask import Flask, request, abort, jsonify, render_template, flash
import spacy



app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27123d441f2b6176a'

nlp = spacy.load('nl')

@app.route('/answer-matching', methods=['POST'])
def match_response():
    if not request.json or not 'predefined-answers' in request.json:
        abort(400)
    if not request.json or not 'user-answer' in request.json:
        abort(400)


    print(request.json['predefined-answers'])
    print(request.json['user-answer'])
    
    answers = [nlp(answer) for answer in request.json['predefined-answers']]
    user_answer = nlp(request.json['user-answer'])
    scores = {}
    for idx, answer in enumerate(answers):
        scores[idx] = answer.similarity(user_answer)
    data = {"scorePerAnswer" : scores}
    result = {"data": data, "success": True, "errorMsg": ""}
    
    return jsonify(result), 200


if __name__ == '__main__':
    # app.run(host="141.85.232.72", debug=False)
    app.run(port=8080)
    