

from flask import Flask, request, render_template
import constants
import preprocessing
from transformers import pipeline

model_dir = "jmilic/roberta-baseline-3"
app = Flask(__name__, instance_relative_config=True)

classifier = pipeline("sentiment-analysis", model=model_dir)


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/infer', methods=['POST'])
def infer():
    print("got message")
    vals = [x for x in request.form.values()]

    first = vals[0]
    second = vals[1]
    third = vals[2]

    row = {'turn1': first, 'turn2': second, 'turn3': third}
    concat_strategy = "roberta"

    input_turns = preprocessing.concat_turns(row, concat_strategy)

    logits = classifier(input_turns)[0]
    take = logits['label']
    index = int(take[len(take) - 1])

    prediction = constants.label2emotion[index]
    return prediction


if __name__ == '__main__':
    print("in main")
    app.run()
