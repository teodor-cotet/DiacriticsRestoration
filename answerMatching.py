from flask import Flask, request, abort, jsonify, render_template, flash, Response
import spacy
from StringKernels import SpectrumStringKernel, IntersectionStringKernel, PresenceStringKernel
import numpy as np

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
    kernels = [PresenceStringKernel, IntersectionStringKernel, SpectrumStringKernel]
    sk_scores = [kernel.compute_kernel_string_listofstrings(
            user_answer.text, 
            [answer.text for answer in answers], 
            3, 7, normalize=True)
        for kernel in kernels]
    sk_scores = [np.mean([score[i] for score in sk_scores]) for i, _ in enumerate(sk_scores[0])]
    spacy_scores = [answer.similarity(user_answer) for answer in answers]
        
    scores = [{"score": (sk + sp) / 2} for sk, sp in zip(sk_scores, spacy_scores)]
    
    data = {"scoresPerOption" : scores}
    result = {"result": data, "success": True, "errorMsg": ""}
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  
    return response, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
    
