from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec, LdaMulticore, LsiModel 
from gensim.corpora import Dictionary
from utils import split_sentences, tokenize_docs
from typing import Dict, List
from os.path import dirname
import numpy as np
from itertools import islice

def train_w2v(inputFilePath: str):
    sentences = [sent for sent in split_sentences(inputFilePath)]
    folder = dirname(inputFilePath)
    print("Starting training...")
    model = Word2Vec(sentences, size=300, window=5, min_count=5, iter=4, workers=16)
    path = folder + "/word2vec.model"
    model.wv.save_word2vec_format(path, binary=False)
    print("Model saved to ", path)

def train_lda(inputFilePath: str):
    corpus = tokenize_docs(inputFilePath)
    id2word = Dictionary(corpus)
    id2word.filter_extremes(no_below=20, no_above=0.1)
    folder = dirname(inputFilePath)
    corpus = [id2word.doc2bow(doc) for doc in corpus]
    print("Starting training...")
    lda = LdaMulticore(corpus, num_topics=300, id2word=id2word)
    path = folder + "/lda.model"
    matrix = np.transpose(lda.get_topics())
    with open(path, "wt", encoding='utf-8') as f:
        f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
        for idx in range(np.size(matrix, 0)):
            f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    print("Model saved to ", path)

def train_lsa(inputFilePath: str):
    corpus = list(tokenize_docs(inputFilePath))
    id2word = Dictionary(corpus)
    id2word.filter_extremes(no_below=20, no_above=0.1)
    folder = dirname(inputFilePath)
    corpus = [id2word.doc2bow(doc) for doc in corpus]
    print("Starting training...")
    lsa = LsiModel(corpus=corpus, id2word=id2word, num_topics=300)
    path = folder + "/lsa.model"
    matrix = np.transpose(lsa.get_topics())
    with open(path, "wt", encoding='utf-8') as f:
        f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
        for idx in range(np.size(matrix, 0)):
            f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    print("Model saved to ", path)


if __name__ == "__main__":
    inputFilePath = "corpora/NL/inl_out.txt"
    print("Loading dataset: ", inputFilePath)
    train_lsa(inputFilePath)
    