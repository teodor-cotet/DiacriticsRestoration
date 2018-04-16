from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec 
from utils import split_sentences
from typing import Dict, List
from os.path import dirname

if __name__ == "__main__":
    inputFilePath = "in/vladcristian_out.txt"
    print("Loading dataset: ", inputFilePath)
    sentences = [sent for sent in split_sentences(inputFilePath)]
    folder = dirname(inputFilePath)
    print("Starting training...")
    model = Word2Vec(sentences, size=300, window=5, min_count=5, iter=4, workers=16)
    path = folder + "/word2vec.model"
    model.wv.save_word2vec_format(path, binary=False)
    print("Model saved to ", path)