import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import librosa
import tensorflow as tf
import traceback
tess_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
app = Flask(__name__)
class QuaternionConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, **kwargs):
        super(QuaternionConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel_real = self.add_weight(name="kernel_real",
                                           shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.filters),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.kernel_imag = self.add_weight(name="kernel_imag",
                                           shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.filters),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(self.filters,),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        input_dim = inputs.shape[-1]
        inputs = tf.reshape(inputs, (-1, 1, 1, input_dim))
        real_part = tf.nn.conv2d(inputs, self.kernel_real, strides=[1, self.strides[0], self.strides[1], 1],
                                 padding=self.padding.upper())
        imag_part = tf.nn.conv2d(inputs, self.kernel_imag, strides=[1, self.strides[0], self.strides[1], 1],
                                 padding=self.padding.upper())
        output = tf.math.subtract(real_part, imag_part) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    @classmethod
    def from_config(cls, config):
        trainable = config.pop('trainable', True)  # Remove 'trainable' from config and set default value if not present
        return cls(trainable=trainable, **config)


# Register the custom layer
tf.keras.utils.get_custom_objects()['QuaternionConv2D'] = QuaternionConv2D

# Now, load the model
model = load_model('my_model.keras')
print("model loaded")


def extract_feature(file_name):
    print("File path:", file_name)
    try:
        # Attempt to load the audio file
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        # Compute STFT and extract magnitude and phase
        stft_representation = librosa.stft(X)
        magnitude = np.abs(stft_representation)
        phase = np.angle(stft_representation)

        # Calculate statistics (mean and standard deviation)
        magnitude_mean = np.mean(magnitude, axis=1)
        magnitude_std = np.std(magnitude, axis=1)
        phase_mean = np.mean(phase, axis=1)
        phase_std = np.std(phase, axis=1)

        # Combine mean and standard deviation to form features
        features = np.hstack((magnitude_mean, magnitude_std, phase_mean, phase_std))
        print(features)
        return features

    except FileNotFoundError:
        print(f"Error: File not found - {file_name}")
        return None
    except Exception as e:
        print(f"Error during feature extraction from {file_name}: {e}")
        return None

@app.route('/')
def upload_page():
    return render_template('uploadpage.html')
# Route for rendering upload form page
@app.route("/loading", methods=['POST'])
@app.route("/prediction", methods=["GET"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify('No file part')

        uploaded_file = request.files['file']

        if uploaded_file is None:
            return jsonify('Uploaded file is None')

        if uploaded_file.filename == '':
            return jsonify('No selected file')

        print("Uploaded filename:", uploaded_file.filename)
        print(type(uploaded_file.filename))
        # Specify the directory where you want to save the uploaded audio files
        upload_directory = 'audio'
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)  # Create the directory if it doesn't exist

        # Generate a unique filename using uuid4 and save the file
        file_path = os.path.join(upload_directory, uploaded_file.filename)
        uploaded_file.save(file_path)
        print("File path:", file_path)  # Print the file path before calling extract_feature

        # Check if the file exists
        if os.path.exists(file_path):
            print("File exists:", file_path)
        else:
            print("File does not exist:", file_path)

        # Extract features from the audio file
        features = extract_feature(file_path)

        # The rest of your code for feature extraction, prediction, and response handling goes here...
    except OSError as e:
        print("OSError:", e)
        traceback.print_exc()  # Print full traceback
    except OSError as e:
        print("OSError:", e)
        return jsonify('Error: {}'.format(str(e)))
    except Exception as e:
        print("Error occurred:", e)
        return jsonify('Error: {}'.format(str(e)))
    if features is None:
        return jsonify('Error: Unable to extract features from the audio file')

    if not hasattr(features, 'shape'):
        return jsonify('Error: Extracted features do not have a shape attribute')

        # Reshape the features to match the input shape of the model
    features = np.expand_dims(features, axis=0)
    try:
        print("Features shape:", features.shape)
        print("Features data type:", features.dtype)
    except Exception as e:
        print("Error occurred while printing features information:", e)
    # Make predictions using the loaded model
    predictions = model.predict(features)

    # Handle predictions and return response
    predictions = predictions.tolist()
    predicted_emotion_index = np.argmax(predictions)
    predicted_emotion = tess_emotions[predicted_emotion_index]
    print("The predicted emotion for the uploaded audio file is:", predicted_emotion)

    return render_template("predictionpage.html", filename=uploaded_file.filename, predicted_emotion=predicted_emotion)

if __name__ == "__main__":
    app.run(debug=True, port=5300)