import numpy as np
import csv
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    saveInCSV = list()

    output = round(prediction[0], 2)

    saveInCSV = list(final_features[0])
    saveInCSV.append(output)
    myFile = open('result_data.csv', 'a')
      
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows([saveInCSV])
    myFile.close()

    if bool(output):
        return  render_template('index.html', prediction_text='Yes, as per given data, there are high chances for having Heart disease')
    else:
        return  render_template('index.html', prediction_text='No, as per given data, there are less chances for having Heart disease')

    

if __name__ == "__main__":
    app.run(debug=True)
