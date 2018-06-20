import spacy
from spacy.lang.ro import Romanian
from typing import Dict, List, Iterable
from nltk import sent_tokenize
import re

# JSON Example localhost:8081/spacy application/json
# {
#     "lang" : "en",
#     "blocks" : ["După terminarea oficială a celui de-al doilea război mondial, în conformitate cu discursul lui W. Churchill (prim ministru al Regatului Unit la acea dată), de la Fulton, s-a declanșat Războiul rece și a apărut conceptul de cortină de fier. Urmare a politicii consecvente de apărare a sistemului economic și politic (implicit a intereslor economice ale marelui capital din lumea occidentală) trupele germane, în calitate de prizonieri, aflate pe teritoriul Germaniei de Vest au fost reînarmate și au constituit baza viitorului Bundeswehr - armata regulată a R.F.G."]
# }

models = {
    'en': 'en',
    'nl': 'nl',
    'fr': 'fr',
    'es': 'es',
    'de': 'de',
    'it': 'it',
    'ro': 'models/model3'
}

normalization = {
    'ro': [
        (re.compile("ş"), "ș"),
        (re.compile("Ş"), "Ș"),
        (re.compile("ţ"), "ț"),
        (re.compile("Ţ"), "Ț"),
        (re.compile("(\w)î(\w)"), "\g<1>â\g<2>")
    ]
}

def convertToPenn(pos: str, lang: str) -> str:
    if lang != 'ro':
        if len(pos) > 2:
            return pos[:2]
        return pos
    pos = pos.lower()
    if pos.startswith("n"):
        return "NN"
    if pos.startwith("v"):
        return "VB"
    if pos.startwith("a"):
        return "JJ"
    if pos.startwith("r"):
        return "RB"
    if pos.startwith("s") or pos.startswith("cs"):
        return "IN"
    if pos.startwith("c"):
        return "CC"
    return ""
    

class SpacyDoc:

    def __init__(self):
        self.ner = spacy.load('xx_ent_wiki_sm')
        # self.romanian = Romanian()
        self.pipelines = {
            lang: spacy.util.get_lang_class(lang)()
            for lang in models
        }
        # for pipeline in self.pipelines.values():
        #     component = pipeline.create_pipe('tagger')   # 3. create the pipeline components
        #     pipeline.add_pipe(component)
        self.loaded_models = {}
        
    def preprocess(self, text: str, lang: str) -> str:
        if lang not in normalization:
            return text
        for pattern, replacement in normalization[lang]:
            text = re.sub(pattern, replacement, text)
        return text

    def get_tokens_lemmas(self, sentences: Iterable, lang: str) -> Iterable:
        if lang not in self.pipelines:
            return None
        pipeline = self.pipelines[lang]
        # sbd = pipeline.create_pipe('sentencizer')
        # pipeline.add_pipe(sbd)
        doc = pipeline.pipe((sent[:1].lower() + sent[1:] for sent in sentences), batch_size=100000, n_threads=16)
        # print([sent.string.strip() for sent in doc.sents])
        # print(len(doc.sents))
        # print("====================")
        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #   token.shape_, token.is_alpha, token.is_stop)
        # print("====================")
        return doc
        # return [(token.text, token.lemma_) for token in doc]

    def tokenize_sentences(self, block: str) -> List[str]:
        return sent_tokenize(block)

    def parse(self, sentence: str, lang: str):
        if lang not in self.loaded_models:
            self.loaded_models[lang] = spacy.load(models[lang])
        model = self.loaded_models[lang]
        doc = model(sentence)
        # print([sent.string.strip() for sent in doc.sents])
        # for chunk in doc.noun_chunks:
        #     print(chunk.text, chunk.root.text, chunk.root.dep_,
        #        chunk.root.head.text)

        # print("********************")
        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #   token.shape_, token.is_alpha, token.is_stop)
        # print("********************")
        # return [(token.text, token.lemma_, token.pos_, token.tag_) for token in doc]
        return doc


    def process(self, doc):
        lang = doc["lang"]
        for block in doc["blocks"]:
            sents = sent_tokenize(block["text"])
            block["sentences"] = list()

            for sent in sents:
                ne = self.ner(sent)
                tokens = self.parse(sent, lang)
                # print(ne)
                # print(pos)
                res_sent = {}
                res_sent["text"] = sent
                res_sent["words"] = []
                # get pos tags 
                for w in tokens:
                    wp = {"text" : w.text}
                    wp["index"] = w.i
                    wp["lemma"] = w.lemma_
                    wp["pos"] = convertToPenn(w.tag_, lang)
                    wp["dep"] = w.dep_
                    wp["ner"] = w.ent_type_
                    wp["head"] = w.head.i
                    res_sent["words"].append(wp)
                # get named entities 
                for ent in [token for token in ne if token.ent_type != 0]:
                    for w in res_sent["words"]:
                        # or (' ' in ent[0] and w["word"] in ent[0])
                        if w["index"] == ent.i:
                            w["ner"] = ent.ent_type_
                block["sentences"].append(res_sent)
        return doc   

if __name__ == "__main__":
    spacyInstance = SpacyDoc()

    sent = """
        După terminarea oficială a celui de-al doilea război mondial, în conformitate cu discursul lui W. Churchill (prim ministru al Regatului Unit la acea dată), de la Fulton, s-a declanșat Războiul rece și a apărut conceptul de cortină de fier. Urmare a politicii consecvente de apărare a sistemului economic și politic (implicit a intereslor economice ale marelui capital din lumea occidentală) trupele germane, în calitate de "prizonieri", aflate pe teritoriul Germaniei de Vest au fost reînarmate și au constituit baza viitorului "Bundeswehr" - armata regulată a R.F.G.

        Pe fondul evenimentelor din 1948 din Cehoslovacia (expulzări ale etnicilor germani, alegeri, reconstrucție economică) apare infiltrarea agenților serviciilor speciale ale S.U.A. și Marii Britanii cu rol de "agitatori". Existând cauza, trupele sovietice nu părăsesc Europa Centrală și de Est cucerită-eliberată, staționând pe teritoriul mai multor state. Aflate pe linia de demarcație dintre cele două blocuri foste aliate, armata sovietică nu a plecat din Ungaria decât după dizolvarea Tratatului de la Varșovia.
        """

    # sent = """
    #     După terminarea oficială a celui de-al doilea război mondial, în conformitate cu discursul lui Churchill, de la Fulton, s-a declanșat Războiul rece și a apărut conceptul de cortină de fier."""

    # print(spacyInstance.get_ner(sent))
    # print(spacyInstance.get_tokens_lemmas(sent))
    for token in spacyInstance.parse(sent, 'ro'):
        print(token.tag_)
    # print(spacyInstance.preprocess("coborî", 'ro'))
