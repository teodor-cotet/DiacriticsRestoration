from gensim.models import Word2Vec 
import spacy
from typing import List, Iterable
from nltk.tokenize import sent_tokenize, WordPunctTokenizer

# nlp = spacy.load('en_core_web_sm')
# nlp.remove_pipe('tagger')
# nlp.remove_pipe('parser')
# nlp.remove_pipe('ner')

def split_sentences(fileName: str) -> Iterable[List[str]]:
    # sentences = []
    tokenizer = WordPunctTokenizer()
    with open(fileName, "rt") as f:
        for line in f.readlines():
            for sent in sent_tokenize(line):
                yield [token for token in tokenizer.tokenize(sent)
                       if token.isalpha and not token == '.']

def tokenize_docs(fileName: str) -> Iterable[List[str]]:
    # sentences = []
    tokenizer = WordPunctTokenizer()
    with open(fileName, "rt") as f:
        for line in f.readlines():
            yield [token for token in tokenizer.tokenize(line)
                if token.isalpha and not token == '.']
            