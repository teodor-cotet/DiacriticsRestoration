from flask import Flask, request, abort, jsonify, render_template, flash, Response
import spacy
from spacy.tokens import Doc     
from readerbench.core.StringKernels import SpectrumStringKernel, IntersectionStringKernel, PresenceStringKernel
import numpy as np
from sklearn.cluster import AffinityPropagation


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

def similarity(a: Doc, b: Doc) -> float:
    kernels = [PresenceStringKernel, IntersectionStringKernel, SpectrumStringKernel]
    sk_scores = [kernel.compute_kernel_string_listofstrings(
            a.text, 
            [b.text], 
            3, 7, normalize=True)
        for kernel in kernels][0]
    sk_score = np.mean([score for score in sk_scores])
    spacy_score = a.similarity(b)
    return (sk_score + spacy_score) / 2

@app.route('/answer-matching', methods=['POST'])
def match_response():
    if not request.json or not 'options' in request.json:
        abort(400)
    if not request.json or not 'input' in request.json:
        abort(400)


    answers = [nlp(answer['text']) for answer in request.json['options'] if len(answer['text'].strip()) > 0]
    user_answer = nlp(request.json['input']['text'])
        
    scores = [{"score": similarity(user_answer, answer)} for answer in answers]
    
    data = {"scoresPerOption" : scores}
    result = {"result": data, "success": True, "errorMsg": ""}
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  
    return response, 200

@app.route('/clustering', methods=['POST'])
def cluster_answers():
    candidates = request.json['candidates']
    answers = request.json['answers']
    minPerCluster = request.json['minPerCluster']
    docs = [nlp(candidate) for candidate in candidates + answers]
    matrix = [[similarity(doc1, doc2) for doc2 in docs] for doc1 in docs]
    clustering = AffinityPropagation(affinity='precomputed')
    clustering.fit(matrix)
    result = {}
    for index, cluster in enumerate(clustering.labels_):
        if cluster not in result:
            result[cluster] = []
        if index < len(candidates):
            result[cluster].append({'text': docs[index].text, 'type': 0})
        else:
            result[cluster].append({'text': docs[index].text, 'type': 1})
        
    response = jsonify({"clusters": [cluster for idx, cluster in result.items()]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  
    return response, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
    
