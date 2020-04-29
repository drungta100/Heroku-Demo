# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# http://localhost:5000/api_predict
model_pk = pickle.load(open("flower-v1.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
        
        #extracting the inputs from html form to variables in python file         
        seplen = request.form['sepal_length']
        sepwid = request.form['sepal_width']
        petlen = request.form['petal_length']
        petwid = request.form['petal_width']

        #converting the string inputs to float inputs so that it can be used for processing
        a = float(seplen)
        b = float(sepwid)
        c = float(petlen)
        d = float(petwid)
        
        #passing the inputs in float form to an array of list in1
        in1 = np.array([[a, b, c, d]])
        
        #prediction is array of dependent variable i.e. the species of the flower
        prediction = model_pk.predict(in1)
        
        text = "The species of the flower is :"
        
        return render_template('index.html', t = text, prediction_text= prediction[0])
        
        
        
if __name__ == "__main__":
    app.run(debug=True)
        
        


























































