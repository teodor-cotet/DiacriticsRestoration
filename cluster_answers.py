import json
from sklearn.cluster import SpectralClustering, AffinityPropagation
import spacy

if __name__ == "__main__":
    dataset = json.load(open("resources/scripts/eit/ai2MatchesGoldenTraining.json", "rt"))
    dataset += json.load(open("resources/scripts/eit/ai2MatchesGoldenValidation.json", "rt"))
    dataset += json.load(open("resources/scripts/eit/ai2MatchesGoldenTesting.json", "rt"))
    questions = {}
    nlp = spacy.load("nl")

    for inst in dataset:
        text = "".join([option["text"] for option in inst["options"]])
        if not text in questions:
            questions[text] = ([option["text"] for option in inst["options"]], [])
        questions[text][1].append(inst["input"]["text"])
    for question, (candidates, inputs) in questions.items():
        docs = [nlp(candidate) for candidate in candidates + inputs]
        matrix = [[doc1.similarity(doc2) for doc2 in docs] for doc1 in docs]
        clustering = AffinityPropagation(affinity='precomputed')
        clustering.fit(matrix)
        result = {}
        for index, cluster in enumerate(clustering.labels_):
            if cluster not in result:
                result[cluster] = []
            if index < len(candidates):
                result[cluster].append("candidate-" + docs[index].text)
            else:
                result[cluster].append("input-" + docs[index].text)
        # print(result)