#provide a way to interact with operating system
import os
#to extract features such as MFCC,melspectrogram and chroma
import librosa
#provides support for multi-dimensional arrays and matrices
import numpy as np
#components of flask web framework
#flask-used to create application instances
#request-used to access incoming request data in flask routes
#jsonify-converts python dictionaries or lists into JSON strings which can be returned as http responses
from flask import Flask, request, jsonify, render_template
#used to save and load trained models
import pickle
#initializes a flask application called app with __name__ represents current module name.usually used when you are not importing anyother modules.
app = Flask(__name__)

# Load trained models which were saved in the other script pav
svm_model = pickle.load(open('svm.pkl', 'rb'))
print("SVM Model loaded successfully.")
mlp_model = pickle.load(open('mlp.pkl', 'rb'))
print("MLP Model loaded successfully.")
knn_model = pickle.load(open('knn.pkl', 'rb'))
print("KNN Model loaded successfully.")
rfm_model = pickle.load(open('rfm.pkl', 'rb'))
print("Random Forest Model loaded successfully.")
nbm_model = pickle.load(open('nb.pkl', 'rb'))
print("Naive Bayes Model loaded successfully.")

# Define your feature extraction function
def extract_feature(file_name):
    try:
        X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)
        mel_spectrogram = np.mean(mel_spectrogram.T, axis=0)
        stft = np.abs(librosa.stft(X))
        # Compute the chroma
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma = np.mean(chroma.T, axis=0)

        # Compute the MFCCs
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)

        # Concatenate all features
        all_features = np.hstack((chroma, mfccs, mel_spectrogram))

        return all_features
    except Exception as e:
        print("Error extracting features:", e)
        return None


# Define your prediction function
def predict_emotion(features, model_name):
    if model_name == 'svm':
        # Use SVM model for prediction
        predicted_emotion = svm_model.predict(features.reshape(1, -1))
    elif model_name == 'mlp':
        # Use MLP model for prediction
        predicted_emotion = mlp_model.predict(features.reshape(1, -1))
    elif model_name == 'knn':
        # Use KNN model for prediction
        predicted_emotion = knn_model.predict(features.reshape(1, -1))
    elif model_name == 'random_forest':
        # Use Random Forest model for prediction
        predicted_emotion = rfm_model.predict(features.reshape(1, -1))
    elif model_name == 'naive_bayes':
        # Use Naive Bayes model for prediction
        if features.shape == (180,):  # Check if the shape is correct
            predicted_emotion = nbm_model.predict(features.reshape(1, -1))
        else:
            predicted_emotion = 'Incorrect feature shape'
    else:
        # Handle unsupported model name
        predicted_emotion = 'Unknown model'

    return predicted_emotion
# Route for rendering the upload page
@app.route('/')
def upload_page():
    return render_template('uploads.html')
#@app.route registers a new route /upload and that it should handle HTTP POST requests
@app.route('/upload', methods=['POST'])
#request.files is a dictionary-like object containing the uploaded file
# New route for rendering prediction results
@app.route('/prediction-results', methods=['GET'])

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
#accesses the value associated with 'file'
#'file' is the name of the file input field in HTML form
    uploaded_file = request.files['file']
#checks whether the filename of uploaded file is an empty string that means no file was selected
    if uploaded_file.filename == '':
        return jsonify({'error': 'No file selected'})
#request.form consists parsed form data submitted with the request
#it has key as the name of the form field and value as values associated with the key
#if there are multiple values submitted to same form field the getlist() gets them all
    model_names = request.form.getlist('model')
    print("Received model names:", model_names)
#example:uploaded_file-file and filename-happy.wav then uploadshappy.wav
#uploads is the directory in which happy.wav is saved
#uploads/happy.wav is saved in uploaded_file(file)
    if model_names:
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)
        # Extract features from the uploaded_file
        features = extract_feature(file_path)
#checks if the features are successfully extracted
#if features are extracted then a dictionary called predictions is initiated to store emotion prediction of each model
        if features is not None:
            predictions = {}
            # Predict emotion for each model
            for model_name in model_names:
                predicted_emotion = predict_emotion(features, model_name)
                # Convert ndarray to list
                #features is in numpy array format since the use of 'np' thus predicted_emotion should also be in numpy format.
                predicted_emotion = predicted_emotion.tolist()
                #in the predictions dictionary model_name is saved with predicted emotion in the form of: {'model_name' : 'happy'}
                predictions[model_name] = predicted_emotion
            return render_template('result.html', emotions=predictions)
        else:
            return jsonify({'error': 'Failed to extract features from the audio file'})
    else:
        return jsonify({'error': 'Model names not provided'})

#This allows you to run the Flask application and debug it directly from the script when it's executed as the main program.
#If the script is imported as a module, this code block will not be executed.
#only if debug=true then the debugger server can be activated by flask development server.
#there is a debugger client which is "pycharm" here.
#which is connected to the debugger server run by the flask application on the local machine.
#the debugger client can be connected to the debugger server using the debugging pin:143-824-119 here.
#then postman sends formdata using localhost and the listens for requests through the port number= 5000
#http//localhost:5000/upload
if __name__ == '__main__':
    app.run(debug=True,port=5000)
