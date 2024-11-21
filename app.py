import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import random 

app = Flask(__name__)

def predict(values, dic):
    # breast_cancer
    if len(values) == 22:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        foo = model.predict(values.reshape(1, -1))[0]
        return foo

    # heart disease
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return foo

    # kidney disease
    elif len(values) == 24:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        foo = model.predict(values.reshape(1, -1))[0]
        return foo

    # liver disease
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        foo = model.predict(values.reshape(1, -1))[0]
        return foo

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except Exception as e:
        print("An error occurred:", e)
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

if __name__ == '__main__':
    app.run(debug = True)