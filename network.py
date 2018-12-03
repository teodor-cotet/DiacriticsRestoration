from typing import List
from os.path import join
import csv
import spacy
from readerbench.core.StringKernels import PresenceStringKernel, IntersectionStringKernel, SpectrumStringKernel
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Lambda
import numpy as np
import json

nlp = spacy.load('nl')

BATCH_SIZE = 32

def read_rb_results(folder: str, sim_type: str) -> List:
    result = []
    with open(join(folder, "file-{}.fix.csv".format(sim_type)), mode="rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        reader.__next__() # skip header
        
        for line in reader:
            text = line[1]
            options = [answer for answer in line[2:8] if len(answer) > 0]
            target = int(line[14]) - 1
            for i, answer in enumerate(options):
                result.append((text, answer, float(line[8 + i]), 1 if target == i else 0))
    return result

def read_sscm(folder: str) -> List:
    with open(join(folder, "SSCM.json"), "rt", encoding='utf-8') as f:
        return [(entry["input"]["text"], option["text"], option["sscmeScore"]) for entry in json.load(f) for option in entry["options"]]

def build_dataset(folder: str, train: bool = True) -> tf.data.Dataset:
    types = ["leacock", "path", "wu", "word2vec"]
    scores = [read_rb_results(folder, sim_type) for sim_type in types]
    kernels = [PresenceStringKernel, IntersectionStringKernel, SpectrumStringKernel]
    x = []
    y = []
    for rb_scores in zip(*scores):
        text1 = rb_scores[0][0]
        text2 = rb_scores[0][1]
        target = rb_scores[0][3]
        sk_scores = [kernel.compute_kernel_string_listofstrings(
                text1, 
                [text2], 
                size, size + 1, normalize=True)[0]
            for kernel in kernels
            for size in range(2, 11, 2)]
        a = nlp(text1)
        b = nlp(text2)
        sk_scores += [score[2] for score in rb_scores]
        sk_scores.append(a.similarity(b))
        x.append(sk_scores)
        y.append(target)
    sscm = [score for answer, option, score  in read_sscm(folder)]
    x = [features + [score] for features, score in zip(x, sscm)]
    if train:
        positive = tf.data.Dataset.from_tensor_slices([features for features, target in zip(x, y) if target == 1])
        negative = tf.data.Dataset.from_tensor_slices([features for features, target in zip(x, y) if target == 0])
        target = tf.data.Dataset.from_tensor_slices([[1]]).repeat()
        positive = positive.repeat()
        positive = positive.shuffle(1024)
        dataset = tf.data.Dataset.zip((positive, negative, target))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1024)
        dataset = dataset.batch(BATCH_SIZE)
    else:
        instances = tf.data.Dataset.from_tensor_slices(x)
        dummy = tf.data.Dataset.from_tensor_slices(x)
        target = tf.data.Dataset.from_tensor_slices([[val] for val in y])
        dataset = tf.data.Dataset.zip((instances, dummy, target))
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.repeat()
    return dataset

def hinge(y_true, y_pred):
    
    return tf.reduce_mean(tf.maximum(0., 0.5 - y_pred))

def build_model() -> tf.keras.Model:
    pos_input = Input((21,))
    neg_input = Input((21,))
    seq = Sequential()
    seq.add(Dense(8, activation='tanh'))
    seq.add(Dense(1, activation='sigmoid'))
    pos_score = seq(pos_input)
    neg_score = seq(neg_input)
    diff = Lambda(lambda pair: pair[0] - pair[1], output_shape=(1,))((pos_score, neg_score))
    model = Model(inputs=(pos_input, neg_input), outputs=(diff, pos_score))
    
    model.compile(optimizer='adam',
        loss={"lambda": hinge, "sequential": lambda x, y: tf.constant(0, dtype=tf.float32)},
        metrics={"sequential": 'accuracy'})
    return model
    
def print_results(folder: str, results: List):
    with open(join(folder, "file-leacock.fix.csv"), mode="rt", encoding="utf-8") as f, open(join(folder, "file-nn.fix.csv"), mode="wt", encoding="utf-8") as out:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        writer = csv.writer(out, delimiter=',', quotechar='"')
        header = reader.__next__() # skip header
        writer.writerow(header)
        current = 0
        for line in reader:
            text = line[1]
            options = [answer for answer in line[2:8] if len(answer) > 0]
            values = results[current:(current + len(options))]
            line[8:(8+len(options))] = values
            current += len(options)
            writer.writerow(line)

def steps(count: int, batch: int) -> int:
    if count % batch == 0:
        return count // batch
    else:
        return count // batch + 1


if __name__ == "__main__":
    # tf.enable_eager_execution()
    train_folder = "resources/Second Experiment/training"
    dev_folder = "resources/Second Experiment/validation"
    pos_train, neg_train, y_train = build_dataset(train_folder, train=True).make_one_shot_iterator().get_next()
    pos_dev, neg_dev, y_dev =  build_dataset(dev_folder, train=False).make_one_shot_iterator().get_next()
    train_len = len([1 for inst in read_rb_results("resources/Second Experiment/training", "leacock") if inst[3] == 0])
    dev_len = len([1 for inst in read_rb_results("resources/Second Experiment/validation", "leacock") if inst[3] == 0])
    print("Training examples: {}".format(train_len))
    print("Validation examples: {}".format(dev_len))
    train_len = steps(train_len, BATCH_SIZE)
    dev_len = steps(dev_len, BATCH_SIZE)
    model = build_model()
    EPOCHS = 42
    BATCH_SIZE = 32
    # tf.logging.set_verbosity(tf.logging.INFO)
    for epoch in range(1, EPOCHS + 1):
        print("Epoch {}: ".format(epoch))
        model.fit([pos_train, neg_train], [y_train, y_train], steps_per_epoch = train_len, validation_data=([pos_dev, neg_dev], [y_dev, y_dev]), validation_steps=dev_len, )
        # print(estimator.evaluate(input_fn=training_input_fn(x_dev, y_dev, shuffle=False), steps=len(x_dev)//BATCH_SIZE))
    train_len = steps(len(read_rb_results("resources/Second Experiment/training", "leacock")), BATCH_SIZE)
    dev_len = steps(len(read_rb_results("resources/Second Experiment/validation", "leacock")), BATCH_SIZE)
    
    _, predictions = model.predict([pos_dev, neg_dev], steps=dev_len)
    print_results(dev_folder, [pred[0] for pred in predictions])
    pos_train, neg_train, y_train = build_dataset(train_folder, train=False).make_one_shot_iterator().get_next()
    
    _, predictions = model.predict([pos_train, neg_train], steps=train_len)
    print_results(train_folder, [pred[0] for pred in predictions])
    