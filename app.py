import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sp_l = float(request.form['sepal length'])
    sp_w = float(request.form['sepal width'])
    p_l = float(request.form['petal length'])
    p_w = float(request.form['petal width'])

    finalFeatures = np.array([[sp_l,sp_w,p_l,p_w]])
    prediction = model.predict(finalFeatures)

    return render_template('index.html', prediction_text='Expected species  $ {}'.format(round(prediction[0])))


if __name__ == "__main__":
    app.run(debug=True)
