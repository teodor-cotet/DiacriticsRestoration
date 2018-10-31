from typing import List
from os.path import join
import csv
import spacy
from readerbench.core.StringKernels import PresenceStringKernel, IntersectionStringKernel, SpectrumStringKernel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

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

def build_dataset(folder: str, shuffle: bool = True) -> tf.data.Dataset:
    types = ["leacock", "path", "wu", "word2vec"]
    scores = [read_rb_results(folder, sim_type) for sim_type in types]
    kernels = [PresenceStringKernel, IntersectionStringKernel, SpectrumStringKernel]
    x = []
    y = []
    for s1, s2, s3, s4 in zip(scores[0], scores[1], scores[2], scores[3]):
        sk_scores = [kernel.compute_kernel_string_listofstrings(
                s1[0], 
                [s1[1]], 
                size, size + 1, normalize=True)[0]
            for kernel in kernels
            for size in range(2, 11, 2)]
        a = nlp(s1[0])
        b = nlp(s1[1])
        sk_scores += [s1[2], s2[2], s3[2], s4[2], a.similarity(b)]
        x.append(sk_scores)
        y.append(s1[3])
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def weighted_loss(y_true, y_pred):
    class_weights = tf.constant([1, 8], dtype=tf.float32)
    weights = tf.nn.embedding_lookup(class_weights, y_true)
    loss = tf.square(tf.to_float(y_true) - y_pred)
    return tf.reduce_mean(loss * weights)

def build_model() -> tf.keras.Model:
    model = Sequential()
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
        loss=weighted_loss)
        # metrics=['accuracy'])
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
    x_train, y_train = build_dataset(train_folder).make_one_shot_iterator().get_next()
    x_dev, y_dev =  build_dataset(dev_folder, shuffle=False).make_one_shot_iterator().get_next()
    train_len = len(read_rb_results("resources/Second Experiment/training", "leacock"))
    dev_len = len(read_rb_results("resources/Second Experiment/validation", "leacock"))
    print("Training examples: {}".format(train_len))
    print("Validation examples: {}".format(dev_len))
    train_len = steps(train_len, BATCH_SIZE)
    dev_len = steps(dev_len, BATCH_SIZE)
    model = build_model()
    EPOCHS = 39
    BATCH_SIZE = 32
    # tf.logging.set_verbosity(tf.logging.INFO)
    for epoch in range(1, EPOCHS + 1):
        print("Epoch {}: ".format(epoch))
        model.fit(x_train, y_train, steps_per_epoch = train_len, validation_data=(x_dev, y_dev), validation_steps=dev_len, )
        # print(estimator.evaluate(input_fn=training_input_fn(x_dev, y_dev, shuffle=False), steps=len(x_dev)//BATCH_SIZE))
    predictions = model.predict(x_dev, steps=dev_len)
    print_results(dev_folder, [pred[0] for pred in predictions])
    x_train, y_train = build_dataset(train_folder, shuffle=False).make_one_shot_iterator().get_next()
    
    predictions = model.predict(x_train, steps=train_len)
    print_results(train_folder, [pred[0] for pred in predictions])
    