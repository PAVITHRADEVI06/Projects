# to find audio files with specific extension in a directory
import glob
# provide a way to interact with operating system
import os
# to save training model and load it later
import pickle
# used for feature extraction such as mfcc,mel spectrogram,chroma
import librosa
# used to measure execution time for certain operations
import time
# to statistically visualize the data and produce confusion plot or figure output you get at the end
import seaborn as sns
# used for numerical operations
import numpy as np
# used when the audio file path and related emotions are categorized into rows and columns
import pandas as pd
# shows progress bar
from tqdm import tqdm

# the file that contains audio files
tess_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


# the function defined to extract mel spectrogram
def extract_mel_spectrogram(file_name):
    # Load audio file using librosa
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)

    # Calculate the mean across all frames
    mean_mel_spectrogram = np.mean(mel_spectrogram.T, axis=0)

    return mean_mel_spectrogram


# the function defined to extract chroma
def extract_chroma(file_name):
    # Load audio file using librosa
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

    # Compute the short-time Fourier transform (STFT) of the input signal
    stft = np.abs(librosa.stft(X))

    # Compute the chromagram
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)

    # Calculate the mean across all frames
    mean_chroma = np.mean(chroma.T, axis=0)

    return mean_chroma


# the function defined to extract mfcc
def extract_mfcc(file_name):
    # Load audio file using librosa
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')

    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)

    # Calculate the mean across all frames
    mean_mfccs = np.mean(mfccs.T, axis=0)

    return mean_mfccs


# extract_feature is the general function used to extract all the features together(mfcc,melspectrogram and chroma)
# only extract_feature function is used in the main part of the script
# stft-short time fourier transform to sample the audio in shorter time frame
# hstack concatenate the previous result with the new result horizontally.
# that means all the features extracted are concatenated in the result and result is returned
# result = (chroma+mfccs+mels)
# file_name is basically x which is individual audio path received from "for x in tqdm(X[0])"
def extract_feature(file_name):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    result = np.array([])
    print(result)

    stft = np.abs(librosa.stft(X))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chromas))
    print(result)

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    print(result)

    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
    result = np.hstack((result, mels))
    print(result)
    # finally all the extracted features are saved in the result
    return result


#
def load_data():
    sound, emo = [], []
    # glob is basically a function which matches the path given in the code to the file's path
    # for example: my file path is "TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav"
    # the glob function just access file directory that has TESS Toronto emotional speech set data
    # and also if it ends with AF_ and .wav
    for file in glob.glob(
            "TESS Toronto emotional speech set data/*AF_*/*.wav"):
        # os.path.basename takes the entire path and reduces it to just "OAF_back_angry.wav"
        file_name = os.path.basename(file)
        # then OAF_back_angry is split by the underscore(_) to OAF,back,angry.wav
        # and by [2] from 0 3rd element is chosen and just angry.wav is taken and by [-4] .wav is removed
        emotion = file_name.split("_")[2][:-4]  # split and remove .wav
        # saves the path "TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav" in sound
        sound.append(file)
        # saves the acquired emotion such as from above example angry in emo
        emo.append(emotion)
    # correlates the list sound to file and emo to emotion
    return {"file": sound, "emotion": emo}


start_time = time.time()
# Trial_dict has the entire file "audio path" and emotion "emotion label"
Trial_dict = load_data()

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

# creates dataframes called X and Y which places the elements in file,emotion into separate rows and a single column
# Trial_dict["file"] only takes audio path from the Trial_dict variable
# Trial_dict["emotion"] only takes emotion labels from the Trial_dict variable
X = pd.DataFrame(Trial_dict["file"])
y = pd.DataFrame(Trial_dict["emotion"])
X.shape, y.shape
# shape property returns a tuple with no. of rows and columns (here it is n no.of rows and 1 column)
# basically counts the no.of times an emotion occurred
y.value_counts()

# X_features = X[0].swifter.progress_bar(enable=True).apply(lambda x: extract_feature(x))
# X_features is initializing an empty list to store the extracted feature
X_features = []

for x in tqdm(X[0]):
    # print(x)
    # X here is the X dataframe, every x is audio file or path in the dataframe X which is passed to extract_feature function
    # the extract_feature function that is already defined above is called
    # and the final output of the function, result is returned and appended for every audio file
    X_features.append(extract_feature(x))

X_features = pd.DataFrame(X_features)
# renaming the label column to emotion
# (y DataFrame remains as y but a label called 'emotion' added at the top of the column)
y = y.rename(columns={0: 'emotion'})
# concatenating the attributes and label into a single dataframe
# (the features for the particular audio file is correlated with the emotions that is already found through the audio file path)
data = pd.concat([X_features, y], axis=1)
# displays 1st five rows of data
data.head()
# we run the script the feature is extracted correlated with emotion stored in data and it is shuffled and stored in TESS_FEATURES.csv file
# then the saved TESS_FEATURES.csv file is loaded
# the extracting of feature and the shuffling happens everytime you run the script
# there is no real use in saving in the file
# reindexing to shuffle the data at random
data = data.reindex(np.random.permutation(data.index))
# Storing shuffled ravdess and tess data to avoid loading again
data.to_csv("TESS_FEATURES.csv")
starting_time = time.time()
data = pd.read_csv("./TESS_FEATURES.csv")
print("data loaded in " + str(time.time() - starting_time) + "ms")

print(data.head())

data.shape

# printing all columns
data.columns

# dropping the column Unnamed: 0 to removed shuffled index
data = data.drop('Unnamed: 0', axis=1)
data.columns

# separating features and target outputs
# In data dataframe there is no way to recognise with column is emotion and which is features.
# So we assigned [0] of emotion as emotion previously using that we can now categorise x to features and y to emotion.
# We drop the column with label emotion to get features and vice versa.
X = data.drop('emotion', axis=1).values
y = data['emotion'].values
print(y)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X.shape, y.shape

np.unique(y)
# sklearn is machine learning library in python
# matplotlib correlates to seaborn used for visualization such as plots and histogram
# Standardscalar is used to normalize data which is basically reducing the range of data
# pipeline is used to describe steps in a single pipeline for easy deploying of models
# confusion matrix used to compared actual value(trained dataset) to predicted value(testing dataset)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# cross validate is used to partition the data into a set of data called fold
# where n no.of fold are used for training and others for validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

# test size is take to be 0.20 which means 20% of dataset is used for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# 5 folds are used with 10 repetitions so there 50 different folds
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
print("******************* svm ************************************************************************")
from sklearn.svm import SVC

# without scaling (i.e, normalizing)
svclassifier = SVC(kernel='rbf')

import time

starting_time = time.time()
# model is trained using x_train and y_train
# x_train is the dataset or features that are used for training
# y_train is the dataset or emotion label that are used for training
svclassifier.fit(X_train, y_train)
print("Trained model in %s ms " % str(time.time() - starting_time))

train_acc = float(svclassifier.score(X_train, y_train) * 100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svclassifier.score(X_test, y_test) * 100)
print("----test accuracy score %s ----" % test_acc)
# setting up StandardScalar for normalizing and
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('SVM', SVC(kernel='rbf'))]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)
# with scaling(i.e, the features are scaled and normalized)
# Fit the pipeline to the training set: svc_scaled
svc_scaled = pipeline.fit(X_train, y_train)
# Assuming X_train is your training data
print("Shape of training data:", X_train.shape)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(svc_scaled.score(X_test, y_test)))
# cross_val_score is a function from scikit-learn to perform cross validation
cv_results2 = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
print(cv_results2)
print("Average:", np.average(cv_results2))

train_acc = float(svc_scaled.score(X_train, y_train) * 100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_scaled.score(X_test, y_test) * 100)
print("----test accuracy score %s ----" % test_acc)
# after the model is trained predictions are made using the x_test which consist the features dataset for testing
# these predictions are emotion predicted for x_test using the patterns absorbed in the training
scaled_predictions = svc_scaled.predict(X_test)
# classification_report basically prints:precision-ratio of true positives to the total no.of positives
# recall-ratio of true positive to actual positive
# F1-score-harmonic mean of precision and recall
# support-no.of actual occurrence
print(classification_report(y_test, scaled_predictions))
# then the y_test dataset that consist the already extracted emotion is compared to the predicted emotion
acc = float(accuracy_score(y_test, scaled_predictions)) * 100
print("----accuracy score %s ----" % acc)
# cm shows the count of correct to incorrect predictions
cm = confusion_matrix(y_test, scaled_predictions)
# the confusion matrix is formed on a subplot ax is the subplot created to hold the confusion matrix cm
ax = plt.subplot()
# a heatmap is used in a plot to represent the data in colour. so it is easy to present correct and incorrect prediction
# annot if true displays the numerical values in the cell
# g means general format it lets the numerical values adapt based on value
# ax means you are telling seaborn to draw the heatmap on ax subplot
sns.heatmap(cm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix SVM');
# the emotion labels are denoted in the x-axis and y-axis from tess_emotion which were already declared above using ticklabels
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
# display the plot
plt.show()
pickle.dump(svc_scaled, open('svm.pkl', 'wb'))
print("******************* mlp   *************************************************************************")
from sklearn.neural_network import MLPClassifier

steps3 = [('scaler', StandardScaler()),
          ('MLP', MLPClassifier())]

pipeline_mlp = Pipeline(steps3)

mlp = pipeline_mlp.fit(X_train, y_train)

print('Accuracy with Scaling: {}'.format(mlp.score(X_test, y_test)))

mlp_train_acc = float(mlp.score(X_train, y_train) * 100)
print("----train accuracy score %s ----" % mlp_train_acc)

mlp_test_acc = float(mlp.score(X_test, y_test) * 100)
print("----test accuracy score %s ----" % mlp_train_acc)

mlp_res = cross_val_score(mlp, X, y, cv=cv, n_jobs=-1)
print(mlp_res)
print("Average:", np.average(mlp_res))

mlp_pred = mlp.predict(X_test)
print(mlp_pred)

print(classification_report(y_test, mlp_pred))

acc_mlp = float(accuracy_score(y_test, mlp_pred)) * 100
print("----accuracy score %s ----" % acc_mlp)

cm_mlp = confusion_matrix(y_test, mlp_pred)

ax = plt.subplot()
sns.heatmap(cm_mlp, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Multi Layer Perceptron)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
pickle.dump(mlp, open('mlp.pkl', 'wb'))
print("******************* KNN ********************************************************************")
from sklearn.neighbors import KNeighborsClassifier

steps4 = [('scaler', StandardScaler()),
          ('KNN', KNeighborsClassifier())]

pipeline_knn = Pipeline(steps4)

knn = pipeline_mlp.fit(X_train, y_train)

print('Accuracy with Scaling: {}'.format(knn.score(X_test, y_test)))

knn_train_acc = float(knn.score(X_train, y_train) * 100)
print("----train accuracy score %s ----" % knn_train_acc)

knn_test_acc = float(knn.score(X_test, y_test) * 100)
print("----test accuracy score %s ----" % knn_train_acc)

knn_res = cross_val_score(knn, X, y, cv=cv, n_jobs=-1)
print(knn_res)
print("Average:", np.average(knn_res))

knn_pred = knn.predict(X_test)
print(knn_pred)

print(classification_report(y_test, knn_pred))

acc_knn = float(accuracy_score(y_test, knn_pred)) * 100
print("----accuracy score %s ----" % acc_knn)

cm_knn = confusion_matrix(y_test, knn_pred)

ax = plt.subplot()
sns.heatmap(cm_knn, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (K Nearest Neighbour)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
pickle.dump(knn, open('knn.pkl', 'wb'))
print("*******************Random forest *********************************************************************")
from sklearn.ensemble import RandomForestClassifier

rfm = RandomForestClassifier()
rfm_score = cross_val_score(rfm, X, y, cv=cv, n_jobs=-1)
print(rfm_score)
print("Average:", np.average(rfm_score))

rfm_res = rfm.fit(X_train, y_train)

rfm_train_acc = float(rfm_res.score(X_train, y_train) * 100)
print("----train accuracy score %s ----" % rfm_train_acc)

rfm_test_acc = float(rfm_res.score(X_test, y_test) * 100)
print("----test accuracy score %s ----" % rfm_test_acc)

rfm_pred = rfm_res.predict(X_test)
print(classification_report(y_test, rfm_pred))

acc = float(accuracy_score(y_test, rfm_pred)) * 100
print("----accuracy score %s ----" % acc)

cm_rfm = confusion_matrix(y_test, rfm_pred)

ax = plt.subplot()
sns.heatmap(cm_rfm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Random Forest)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
pickle.dump(rfm, open('rfm.pkl', 'wb'))
print("******************* Gaussian NB ***********************************************************************")
from sklearn.naive_bayes import GaussianNB

nbm = GaussianNB().fit(X_train, y_train)

nbm_train_acc = float(nbm.score(X_train, y_train) * 100)
print("----train accuracy score %s ----" % nbm_train_acc)

nbm_test_acc = float(nbm.score(X_test, y_test) * 100)
print("----test accuracy score %s ----" % nbm_train_acc)

nbm_score = cross_val_score(nbm, X, y, cv=cv, n_jobs=-1)
print(nbm_score)
print("Average:", np.average(nbm_score))

nbm_pred = nbm.predict(X_test)
print(nbm_pred)

print(classification_report(y_test, nbm_pred))

acc_nbm = float(accuracy_score(y_test, nbm_pred)) * 100
print("----accuracy score %s ----" % acc_nbm)

cm_nbm = confusion_matrix(y_test, nbm_pred)

ax = plt.subplot()
sns.heatmap(cm_nbm, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Gaussian Naive Bayes)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()
# gaussian naive bayes is saved using pickle library in a file named nb.pkl
pickle.dump(nbm, open('nb.pkl', 'wb'))

# the machine learning methods:
# SVM - SUPPORT VECTOR MACHINE
# MLP - MULTI LAYER PERCEPTRON
# KNN - K NEAREST NEIGHBOUR
# RANDOM FOREST
# NBM - GAUSSIAN NAIVE BAYES
# confusion matrix is basically if we have 100 samples and when we train them out of them 20 is angry and 20 is happy
# and when the test dataset is run then the samples for angry is predicted to be 15 and happy to be 14
# then 5 samples were wrongly predicted for angry and 6 for happy
# if you want to the actual value then check out the support column in the table in the command prompt
