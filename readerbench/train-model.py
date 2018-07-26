from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec, LdaMulticore, LsiModel 
from gensim.corpora import Dictionary
from core.utils import split_sentences, tokenize_docs, load_docs
from typing import Dict, List, Iterable
from os.path import dirname
import numpy as np
from itertools import islice
from core.spacy_parser import SpacyParser
from math import log
import time

def train_w2v(sentences: List, outputFolder: str):
    print("Starting training...")
    model = Word2Vec(sentences=sentences, size=300, window=5, min_count=5, iter=4, workers=16)
    path = outputFolder + "/word2vec.model"
    model.wv.save_word2vec_format(path, binary=False)
    print("Model saved to ", path)

def train_lda(docs: List, outputFolder: str):
    docs = list(docs)
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


def train_lsa(docs: Iterable, outputFolder: str):
    docs = list(docs)
    id2word = Dictionary(docs)
    id2word.filter_extremes(no_below=20, no_above=0.1, keep_n=1000000)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    corpus = log_entropy_norm(corpus)
    print("Starting training...")
    lsa = LsiModel(corpus=corpus, id2word=id2word, num_topics=300)
    path = outputFolder + "/lsa.model"
    lsa.save(outputFolder + "/lsa.bin")
    matrix = np.transpose(lsa.get_topics())
    with open(path, "wt", encoding='utf-8') as f:
        f.write("{} {}\n".format(np.size(matrix, 0), np.size(matrix, 1)))
        for idx in range(np.size(matrix, 0)):
            f.write(id2word[idx] + " " + " ".join([str(x) for x in matrix[idx]]) + "\n")
    print("Model saved to ", path)

def preprocess(parser: SpacyParser, folder: str, lang: str, split_sent: bool = True, only_dict_words: bool = False) -> Iterable[List]:
    if only_dict_words:
        test = lambda x: not x.is_oov
    else:
        test = lambda x: True
    for doc in load_docs(folder):
        result = []
        doc = parser.preprocess(doc, lang)
        for tokens in parser.get_tokens_lemmas(parser.tokenize_sentences(doc), lang):
            if len(tokens) == 0:
                continue
            tokens = [token.lemma_.lower() for token in tokens if not token.is_stop and token.is_alpha and test(token)]
            if split_sent:
                yield tokens
            else:
                result += tokens
        if not split_sent:
            yield result

if __name__ == "__main__":
    inputFolder= "resources/corpora/FR/Le Monde"
    parser = SpacyParser()
    print("Loading dataset...")
    sentences = preprocess(parser, inputFolder, 'fr', split_sent=False, only_dict_words=True)
    # train_w2v(sentences, inputFolder)
    # train_lsa(sentences, inputFolder)
    train_lda(sentences, inputFolder)
    # model = LsiModel.load(inputFolder + "/lsa.bin")
    # print(model.projection.s)
    # print(model.get_topics())