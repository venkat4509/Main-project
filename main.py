import pickle
from flask import Flask, render_template, request

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
    return render_template('index.html', prediction_test=f'Final Outcome of the patient is: {output}')


#Heart
@app.route('/predictHeart', methods=['POST'])
def predictHeart():

    # get the input features as a list of strings
    featuresHeart = [x for x in request.form.values()]

    # convert the input features to a numpy array and scale them
    input_featuresHeart = scalerheart.transform([featuresHeart])

    # make the prediction using the loaded model
    predictionHeart = heart.predict(input_featuresHeart)

    outputHeart = predictionHeart[0]
    return render_template('index.html', prediction_testheart=f'Final Outcome of the patient is: {outputHeart}')


#Breastcancer
@app.route('/predictbreast', methods=['POST'])
def predictbreast():

    # get the input features as a list of strings
    featuresBreast = [x for x in request.form.values()]

    # convert the input features to a numpy array and scale them
    input_featuresBreast = scalerbreast.transform([featuresBreast])

    # make the prediction using the loaded model
    predictionBreast = breast.predict(input_featuresBreast)

    output = predictionBreast[0]
    return render_template('index.html', prediction_testbreast=f'Final Outcome of the patient is: {output}')

if __name__ == '__main__':
    app.run(debug=True)
