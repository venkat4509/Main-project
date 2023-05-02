import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# load the saved model
model = pickle.load(open('diabetes_model.pkl', 'rb'))
# load the saved StandardScaler object

scaler = pickle.load(open('scaler.pkl', 'rb'))

app.template_folder = 'template'

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)
