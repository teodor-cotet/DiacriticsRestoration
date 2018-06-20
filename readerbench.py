from flask import Flask, request, abort, jsonify, render_template, flash, Response
import spacy
from spacy_doc import SpacyDoc

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27123d441f2b6176a'
spacyInstance = SpacyDoc()


@app.route('/test', methods=['POST'])
def handle_get():
    # if not request.json:
    #     abort(400)

    print(request.json)

    result = {"doc": {"blocks": [{"block": "Ana are mere", sentences: [{}]}]}}
    response = jsonify(result)
    return response, 200

@app.route('/spacy', methods=['POST'])
def create_spacy_doc():
    doc = spacyInstance.process(request.json['doc'])
    response = jsonify({'doc': doc})

    return response, 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=False)
