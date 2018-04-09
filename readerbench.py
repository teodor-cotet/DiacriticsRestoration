from flask import Flask, request, abort, jsonify, render_template, flash
from flask_cors import CORS, cross_origin
import spacy



app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27123d441f2b6176a'
# app.config['CORS_HEADERS'] = 'Content-Type'
# CORS(app)
# cors = CORS(app, resources={r"/answer-matching": {"origins": "*"}})


nlp = spacy.load('nl')

@app.route('/', methods=['GET'])
def main_render():
	return render_template('main.html')

@app.route('/answer-matching', methods=['POST', 'OPTIONS'])
# @cross_origin(origin='*')
def match_response():
    if not request.json or not 'predefined-answers' in request.json:
        abort(400)
    if not request.json or not 'user-answer' in request.json:
        abort(400)


    print(request.json['predefined-answers'])
    print(request.json['user-answer'])
    
    answers = [nlp(answer) for answer in request.json['predefined-answers'] if len(answer.strip()) > 0]
    user_answer = nlp(request.json['user-answer'])
    scores = {}
    for idx, answer in enumerate(answers):
        scores[idx] = answer.similarity(user_answer)
    data = {"scorePerAnswer" : scores}
    result = {"data": data, "success": True, "errorMsg": ""}
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  
    return response, 200


if __name__ == '__main__':
    app.run(host="141.85.232.72", port=8081, debug=False)
    # app.run(port=8080)
    
