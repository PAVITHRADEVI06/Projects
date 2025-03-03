import glob
import os
import librosa
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

tess_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    stft_representation = librosa.stft(X)
    magnitude = np.abs(stft_representation)
    phase = np.angle(stft_representation)

    # Calculate mean and standard deviation of magnitude and phase
    magnitude_mean = np.mean(magnitude, axis=1)
    magnitude_std = np.std(magnitude, axis=1)
    phase_mean = np.mean(phase, axis=1)
    phase_std = np.std(phase, axis=1)

    # Combine mean and standard deviation to form quaternion features
    features = np.hstack((magnitude_mean, magnitude_std, phase_mean, phase_std))
    print(features)
    return features

def load_data():
    sound, emo = [], []
    for file in glob.glob("TESS Toronto emotional speech set data/*AF_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = file_name.split("_")[2][:-4]  # split and remove .wav
        sound.append(file)
        emo.append(emotion)
    return {"file": sound, "emotion": emo}

# Load data
data_dict = load_data()
X_paths = pd.DataFrame(data_dict["file"])
y = pd.DataFrame(data_dict["emotion"])

# Extract features
X_features = []
for file_path in X_paths[0]:
    features = extract_feature(file_path)
    X_features.append(features)
X_features = pd.DataFrame(X_features)
y = y.rename(columns={0: 'emotion'})
# Concatenate features and labels
data = pd.concat([X_features, y], axis=1)
data = data.reindex(np.random.permutation(data.index))
# Storing shuffled ravdess and tess data to avoid loading again
data.to_csv("TESS_FEATURES.csv")

starting_time = time.time()
data = pd.read_csv("./TESS_FEATURES.csv")
print("Data loaded in " + str(time.time() - starting_time) + " seconds")

data = data.drop('Unnamed: 0', axis=1)

# Separate features and labels
X = data.drop(columns=['emotion']).values
y = data['emotion'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("shape of X_train:", X_train.shape)
print("shape of X_test:", X_test.shape)
########for neural network for better understanding please refer to notes##########
# Define QuaternionConv2D layer
# tensor flow is used to access keras
# keras is a neural network library used to define model in high-level building like layers
# defines a custom layer in keras called quaternionConv2D
# __init__ is a constructor method and it initializes the attributes in its parameters
# attributes - filter/kernel: no. of output filters in the convolution
# kernel_size: size of the convolutional kernel(a tuple of two integers)
# kernel is basically square-shaped matrix or tensor(in high dimension) the values in it is the weights and parameters which are learned during training process
# strides: step size or the no. of steps the kernel moves about in the input image
# padding: without the output feature map will have less spatial dimension than the input image. so keep it the same padding is added.
# activation is for activation function if the activation is 'none' no activation fucntion will be applied
class QuaternionConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None):
        super(QuaternionConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)

# based on the shape of the input the weights are initialized
# input_dim extracts the no.of input channels(feature maps) from the input_shape
# self.filters: no.of filters applied
# glorot_uniform initializes weights randomly helps gradients from vanishing
# trainable=true indicates that weights are trainable and can be updated during training through backpropagation
# initializer='zeros' initializes bias value to zeros, it is a common practice for biases
# self.bias - shape(self.filters) sets the shape to no.of output filters
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
        input_dim = inputs.shape[-1]  # Get the number of channels from input tensor
        inputs = tf.reshape(inputs, (-1, 1, 1, input_dim))
        real_part = tf.nn.conv2d(inputs, self.kernel_real, strides=[1, self.strides[0], self.strides[1], 1],
                                 padding=self.padding.upper())
        imag_part = tf.nn.conv2d(inputs, self.kernel_imag, strides=[1, self.strides[0], self.strides[1], 1],
                                 padding=self.padding.upper())
        output = tf.math.subtract(real_part, imag_part) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),  # Input layer
    QuaternionConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels for training data
y_train_encoded = label_encoder.fit_transform(y_train)

# Transform labels for test data
y_test_encoded = label_encoder.transform(y_test)
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define and compile your model
# (Assuming the model definition and compilation are done before this step)

# Train the model with the encoded labels
model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print("Test Accuracy:", test_acc)
# Make predictions on the test set
predictions = model.predict(X_test)
# Convert probabilities to class labels
# Convert probabilities to class labels for predictions
# Convert probabilities to class labels for predictions
# Convert probabilities to class labels for predictions
# Convert probabilities to class labels for predictions
predicted_labels = np.argmax(predictions, axis=1)

# Convert predicted labels back to original labels
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Convert y_test_encoded back to original labels
y_test_labels = label_encoder.inverse_transform(y_test_encoded)

# Generate the confusion matrix
cm = confusion_matrix(y_test_labels, predicted_labels)

# Generate the classification report
print(classification_report(y_test_labels, predicted_labels, target_names=tess_emotions))

# the confusion matrix is formed on a subplot ax is the subplot created to hold the confusion matrix cm
ax = plt.subplot()
# a heatmap is used in a plot to represent the data in colour. so it is easy to present correct and incorrect prediction
# annot if true displays the numerical values in the cell
# g means general format it lets the numerical values adapt based on value
# ax means you are telling seaborn to draw the heatmap on ax subplot
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=tess_emotions, yticklabels=tess_emotions)
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
# display the plot
plt.show()
# You can then evaluate the accuracy of your predictions
from sklearn.metrics import accuracy_score
# Define the directory where you want to save the model
from keras.saving import save_model
save_model(model, 'my_model.keras')
print("Model saved successfully.")

