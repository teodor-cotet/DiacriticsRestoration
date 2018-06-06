from flask import Flask, request, abort, jsonify, render_template, flash, Response
import spacy

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27123d441f2b6176a'

nlp = spacy.load('nl')

@app.route('/answer-matching', methods=['OPTIONS'])
def handle_options():
    response = Response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  
    return response, 200


@app.route('/answer-matching', methods=['POST'])
def match_response():
    if not request.json or not 'options' in request.json:
        abort(400)
    if not request.json or not 'input' in request.json:
        abort(400)


    answers = [nlp(answer['text']) for answer in request.json['options'] if len(answer['text'].strip()) > 0]
    user_answer = nlp(request.json['input']['text'])
    
    scores = [{"score": answer.similarity(user_answer)} for answer in answers]
    
    data = {"scoresPerOption" : scores}
    result = {"result": data, "success": True, "errorMsg": ""}
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  
    return response, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=False)
    
