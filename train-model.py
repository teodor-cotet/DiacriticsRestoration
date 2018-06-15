from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec, LdaMulticore, LsiModel 
from gensim.corpora import Dictionary
from utils import split_sentences, tokenize_docs, load_docs
from typing import Dict, List, Iterable
from os.path import dirname
import numpy as np
from itertools import islice
from spacy_doc import SpacyDoc
from math import log

def train_w2v(sentences: List, outputFolder: str):
    print("Starting training...")
    model = Word2Vec(sentences=sentences, size=300, window=5, min_count=5, iter=4, workers=16)
    path = outputFolder + "/word2vec.model"
    model.wv.save_word2vec_format(path, binary=False)
    print("Model saved to ", path)

def train_lda(docs: List, outputFolder: str):
    id2word = Dictionary(docs)
    id2word.filter_extremes(no_below=20, no_above=0.1, keep_n=1000000)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    print("Starting training...")
    lda = LdaMulticore(corpus, num_topics=300, id2word=id2word)
    path = outputFolder + "/lda.model"
    matrix = np.transpose(lda.get_topics())
    with open(path, "wt", encoding='utf-8') as f:
        f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
        for idx in range(np.size(matrix, 0)):
            f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    print("Model saved to ", path)

def log_entropy_norm(corpus: List) -> List:
    result = []
    for doc in corpus:
        if len(doc) == 0:
            result.append(doc)
            continue
        total = sum(count for idx, count in doc)
        entropy = sum((count / total) * log(count / total) for idx, count in doc)
        entropy = 1 + (entropy / log(len(corpus)))
        result.append([(idx, log(1 + count) * entropy) for idx, count in doc])
    return result


def train_lsa(docs: List, outputFolder: str):
    id2word = Dictionary(docs)
    id2word.filter_extremes(no_below=20, no_above=0.1, keep_n=1000000)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    corpus = log_entropy_norm(corpus)
    print("Starting training...")
    lsa = LsiModel(corpus=corpus, id2word=id2word, num_topics=300)
    path = outputFolder + "/lsa.model"
    matrix = np.transpose(lsa.get_topics())
    with open(path, "wt", encoding='utf-8') as f:
        f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
        for idx in range(np.size(matrix, 0)):
            f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    print("Model saved to ", path)

def preprocess(parser: SpacyDoc, folder: str, lang: str, split_sent: bool = True) -> Iterable[List]:
    for doc in load_docs(folder):
        result = []
        for sent in parser.tokenize_sentences(doc):
            tokens = parser.get_tokens_lemmas(sent, lang)
            tokens = [token.lemma_.lower() for token in tokens if not token.is_stop and token.is_alpha]
            if split_sent:
                yield tokens
            else:
                result += tokens
        if not split_sent:
            yield result

if __name__ == "__main__":
    inputFolder= "RO/ReadME"
    parser = SpacyDoc()
    print("Loading dataset...")
    sentences = list(preprocess(parser, inputFolder, 'ro', split_sent=False))
    # train_w2v(sentences, inputFolder)
    # train_lsa(sentences, inputFolder)
    train_lda(sentences, inputFolder)
    
    