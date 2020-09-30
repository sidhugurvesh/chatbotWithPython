from flask import Flask, render_template, request
from nltk.stem.lancaster import LancasterStemmer
import tflearn as tf
import tensorflow as tensor
import json
import numpy as np
import nltk
nltk.download('punkt')
stemmer = LancasterStemmer()

# imports
app = Flask(__name__)

with open("qa.json") as file:
    data = json.load(file)

words = []
classes = []
docs_x = []
docs_y = []

for i in data["objects"]:
    for j in i["patterns"]:
        wrds = nltk.word_tokenize(j)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(i["tag"])

    if i["tag"] not in classes:
        classes.append(i["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

classes = sorted(classes)

training = []
output = []

out_empty = [0 for _ in range(len(classes))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[classes.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

tensor.reset_default_graph()

net = tf.input_data(shape=[None, len(training[0])])
net = tf.fully_connected(net, 8)
net = tf.fully_connected(net, 8)
net = tf.fully_connected(net, len(output[0]), activation="softmax")
net = tf.regression(net)

model = tf.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tf")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


app.app_context().push()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
# function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    # print("Welcome. If you are a recruiter, hiring Manager/Team member, you made the right choice!!")
    # print("You can easily conduct a virtual interview. I am going to try my best to answer the questions accordingly")
    # print("Warning!! I am still under development.")
    # print("Example question: type experience in the chat")

    result = model.predict([bag_of_words(userText, words)])
    result_index = np.argmax(result)
    tag = classes[result_index]
    responses = 0
    if result[0][result_index] > 0.60:
        for i in data["objects"]:
            if i['tag'] == tag:
                responses = i['responses']

    else:
        responses = "I am still under development. Try keywords like resume, github etc"
        # print(result)
        # print(result[0][result_index])

    return responses
