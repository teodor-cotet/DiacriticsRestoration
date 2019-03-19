from nltk.corpus import wordnet as wn
from typing import List, Tuple, Callable

lang_dict = {
    'en': 'eng',
    'nl': 'nld',
    'fr': 'fra',
    'ro': 'ron',
    'it': 'ita'
}

def compute_similarity(a: str, b:str, lang: str, sim: Callable[[wn.Synset, wn.Synset], float]) -> float:
    if lang not in lang_dict:
        return 0
    lang = lang_dict[lang]
    return min([
        sim(syn_a, syn_b)
        for syn_a in wn.synsets(a, lang=lang)
        for syn_b in wn.synsets(b, lang=lang)],
        default=0)

def path_similarity(a: str, b: str, lang: str) -> float:
    return compute_similarity(a, b, lang, wn.path_similarity)

def leacock_chodorow_similarity(a: str, b: str, lang: str) -> float:
    return compute_similarity(a, b, lang, wn.lch_similarity)

def wu_palmer_similarity(a: str, b: str, lang: str) -> float:
    return compute_similarity(a, b, lang, wn.wup_similarity)

if __name__ == "__main__":
    print(path_similarity('hond', 'kat', 'nl'))