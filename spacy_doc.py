import spacy
from spacy.lang.ro import Romanian
from typing import Dict, List
from nltk import sent_tokenize

class SpacyDoc:

    def __init__(self):
        self.ner = spacy.load('xx_ent_wiki_sm')
        self.romanian = Romanian()
        self.postag = spacy.load('models/model3')

    def get_ner(self, sentence: str):
        doc = self.ner(sentence)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_tokens_lemmas(self, sentence: str):
        sbd = self.romanian.create_pipe('sentencizer')
        self.romanian.add_pipe(sbd)
        doc = self.romanian(sentence)
        print([sent.string.strip() for sent in doc.sents])
        print(len(doc.sents))
        # print("====================")
        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #   token.shape_, token.is_alpha, token.is_stop)
        # print("====================")
        return [(token.text, token.lemma_) for token in doc]

    def get_pos_tags(self, sentence: str):
        doc = self.postag(sentence)
        # print([sent.string.strip() for sent in doc.sents])
        for chunk in doc.noun_chunks:
            print(chunk.text, chunk.root.text, chunk.root.dep_,
               chunk.root.head.text)

        # print("********************")
        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #   token.shape_, token.is_alpha, token.is_stop)
        # print("********************")
        return [(token.text, token.lemma_, token.pos_, token.tag_) for token in doc]


    def process(self, blocks : List):
        result = list()

        for block in blocks:
            res = {"block" : block}
            sents = sent_tokenize(block)
            res["sentences"] = list()

            for sent in sents:
                ne = self.get_ner(sent)
                pos = self.get_pos_tags(sent)
                print(ne)
                print(pos)
                res_sent = {}
                res_sent["sentence"] = sent
                res_sent["ner"] = ne
                res_sent["words"] = []
                # get pos tags 
                for w in pos:
                    wp = {"word" : w[0]}
                    wp["lemma"] = w[1]
                    wp["tag"] = w[3]
                    res_sent["words"].append(wp)
                # get named entities 
                for ent in ne:
                    for w in res_sent["words"]:
                        # or (' ' in ent[0] and w["word"] in ent[0])
                        if w["word"] == ent[0]:
                            w["ent"] = ent[1]


                res["sentences"].append(res_sent)
            result.append(res)

        return result

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
    # spacyInstance.get_pos_tags(sent)
    print(spacyInstance.process([sent]))
