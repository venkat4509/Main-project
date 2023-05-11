import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# load the saved model
model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart = pickle.load(open('heart_model.pkl', 'rb'))
breast = pickle.load(open('breastcancer_model.pkl', 'rb'))

# load the saved StandardScaler object
scaler = pickle.load(open('scaler.pkl', 'rb'))
scalerheart = pickle.load(open('scalerheart.pkl', 'rb'))
scalerbreast = pickle.load(open('scalerbreast.pkl', 'rb'))

app.template_folder = 'template'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Diabetes.html')
def diabetes():
    return render_template('Diabetes.html')

@app.route('/Heart.html')
def heartdisease():
    return render_template('Heart.html')

@app.route('/BreastCancer.html')
def breastCancer():
    return render_template('BreastCancer.html')

#Diabetes
@app.route('/predict', methods=['POST'])
def predict():

    # get the input features as a list of strings
    features = [x for x in request.form.values()]

    # convert the input features to a numpy array and scale them
    input_features = scaler.transform([features])

    # make the prediction using the loaded model
    prediction = model.predict(input_features)

    output = prediction[0]
    return render_template('Diabetes.html', prediction_test=f'Final Outcome of the patient is: {output}')


#Heart
@app.route('/predictHeart', methods=['POST'])
def predictHeart():

    # get the input features as a list of strings
    featuresHeart = [x for x in request.form.values()]

    # convert the input features to a numpy array and scale them
    input_featuresHeart = scalerheart.transform([featuresHeart])

    # make the prediction using the loaded model
    predictionHeart = int(np.round(heart.predict(input_featuresHeart))[0])

    outputHeart = predictionHeart
    return render_template('Heart.html', prediction_testheart=f'Final Outcome of the patient is: {outputHeart}')


#Breastcancer
@app.route('/predictbreast', methods=['POST'])
def predictbreast():

    # get the input features as a list of strings
    featuresBreast = [x for x in request.form.values()]

    # convert the input features to a numpy array and scale them
    input_featuresBreast = scalerbreast.transform([featuresBreast])

    # make the prediction using the loaded model
    predictionBreast = int(np.round(breast.predict(input_featuresBreast))[0])

    output = predictionBreast
    return render_template('BreastCancer.html', prediction_testbreast=f'Final Outcome of the patient is: {output}')

if __name__ == '__main__':
    app.run(debug=True)
