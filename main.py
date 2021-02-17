from flask import Flask, jsonify, request
from typing import Literal

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import xgboost as xgb


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]

## TODO: these pickled models and descriptions should be read from a model service. Models stored in S3
idxToCondition = json.load(open('idxToCondition.json'))

model2 = pickle.load(open('v_0_1.pkl', "rb"))

tf2 = pickle.load(open("tfidf-descriptions.pkl", 'rb'))
tfidf2 = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 3), stop_words='english',
                        vocabulary = tf2.vocabulary_)



def predict(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """
    f1 = tfidf2.fit_transform([description]).toarray()
    ypred = model2.predict(xgb.DMatrix(f1))
    return idxToCondition[str(np.argmax(ypred))]


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)

    prediction = predict(data["description"])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run()


