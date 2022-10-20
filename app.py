import re
from flask import Flask,request,jsonify,url_for,render_template
import pickle
import numpy as np 
import pandas as pd

app = Flask(__name__)
logistic_model = pickle.load(open("fireprediction.pkl","rb"))
scale = pickle.load(open("scaling.pkl","rb"))
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prdict",methods = ["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scale.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output  = logistic_model.predict(final_input)[0]
    return render_template("home.html",prediction_text = "there is {}".format(output))

if __name__ =="__main__":
    app.run()
