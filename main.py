# Main modules import
import os
from datetime import datetime
# Needed parts - import
from pathlib import Path

import librosa
# Side modules import
import numpy as np
import pandas as pd
from IPython.display import display
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Reshape, LSTM, Conv2D, MaxPooling2D, Bidirectional, \
    BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from evaluation import evaluate

# Python local version

# Root directory for whole project and files
root_dir = Path.cwd()

# Train Set directory and listing of all available recordings
train_set_dir = os.path.join(Path.cwd(), "Recordings")
train_set_list = os.listdir(train_set_dir)

# Evaluation Set directory and listing of all available recordings
eval_set_dir = os.path.join(Path.cwd(), "Eval_Recordings")
eval_set_list = os.listdir(eval_set_dir)

# Directories for saving CSV file with results for evaluation and model from current run
CSV_dir = os.path.join(Path.cwd(), "CSV")
model_dir = os.path.join(Path.cwd(), "Model")

# Colab with Drive version

# Mounting Drive for project processing
# drive.mount('/content/gdrive/')

# Root directory of whole project
# root_dir = '/content/gdrive/MyDrive/Speech_Recognition'

# Directory for saving the models
# model_dir = '/content/gdrive/MyDrive/Speech_Recognition/Model'

# Directory for saving CSV file with results from evaluation
# CSV_dir = '/content/gdrive/MyDrive/Speech_Recognition/CSV'

# Directories with training and evaluation data
# train_set_dir = '/content/gdrive/MyDrive/Speech_Recognition/Recordings'
# train_set_list = os.listdir(train_set_dir)
# eval_set_dir = '/content/gdrive/MyDrive/Speech_Recognition/Eval_Recordings'
# eval_set_list = os.listdir(eval_set_dir)

# Starting in root directory
os.chdir(root_dir)

# Parameters for comfortable changes later
win_length = 256                                # Length of MFCC window
n_batch_size = 20                               # Batch Size
n_epoch = 100                                   # Number of Epoch's to perform
n_mfcc = 20                                     # Number of MFCC coefficients

add_display = 1                                 # Display of data during training (consistent with verbose)
loss_function = 'categorical_crossentropy'      # Type of loss function (imported : binary_crossentropy, categorical_crossentropy)
optimizer = 'Adam'                              # Type of optimizer (included : Adam and SGD with lr=0.001)
metric_type = 'accuracy'                        # Type of testing metric
test_percentage = 0.2                           # How much of data will be selected to test the training

sr = 16000                                      # Sampling Rate
n_dct_type = 2                                  # Type of DCT used in MFCC calculation

# Preprocessing of training data set with data augmentation (due to small database)
os.chdir(train_set_dir)

audio_train_database = pd.DataFrame()
audio_name = []
audio_data = []
audio_label = []
for count, audio in enumerate(train_set_list):
    # Classic recordings
    data, fs = librosa.load(audio, sr=None)
    audio_data.append(data)
    audio_name.append(audio)
    audio_label.append(audio[6])

    # Time shifted recordings
    time_shifted = np.roll(data, (sr, 10))
    audio_data.append(time_shifted)
    audio_name.append(audio + str(" [t-s]"))
    audio_label.append(audio[6])

    # Pitch shifted recordings with time shift (+1)
    pitch_shift = librosa.effects.pitch_shift(time_shifted, sr, n_steps=1)
    audio_data.append(pitch_shift)
    audio_name.append(audio + str(" [tp - 1]"))
    audio_label.append(audio[6])

    # Pitch shifted recordings (+4)
    pitch_shift = librosa.effects.pitch_shift(data, sr, n_steps=4)
    audio_data.append(pitch_shift)
    audio_name.append(audio + str(" [p - 4]"))
    audio_label.append(audio[6])

    # Pitch shifted recordings with time shift (-2)
    pitch_shift = librosa.effects.pitch_shift(time_shifted, sr, n_steps=-2)
    audio_data.append(pitch_shift)
    audio_name.append(audio + str(" [tp - (-2)]"))
    audio_label.append(audio[6])

    # Pitch shifted recordings (-5)
    pitch_shift = librosa.effects.pitch_shift(data, sr, n_steps=-5)
    audio_data.append(pitch_shift)
    audio_name.append(audio + str(" [p - (-5)]"))
    audio_label.append(audio[6])

    # Noised recordings
    noised = data + 0.008*np.random.normal(0,1,len(data))
    audio_data.append(noised)
    audio_name.append(audio + str(" [n]"))
    audio_label.append(audio[6])

    # Time stretched recordings (0.3)
    stretched = librosa.effects.time_stretch(data, 0.3)
    audio_data.append(stretched)
    audio_name.append(audio + str(" [str]"))
    audio_label.append(audio[6])

audio_train_database["Name"] = audio_name
audio_train_database["Label"] = audio_label
audio_train_database["Data"] = audio_data

display(audio_train_database)

# Calculation of mel-spectrogram's is based on search for longest recording in evaluation base and manipulating other recordings (by zero-padding or cutting) to same sizes.
# Preprocessing of evaluation database - calculating mel-spectrogram's [with normalization and reshape to proper input shape]

# In this version the MFCC is used

os.chdir(root_dir)

eval_set_base = pd.DataFrame()

os.chdir(eval_set_dir)
MFCC_list = []
delta_list = []
records = []
for count, audio in enumerate(eval_set_list):
    data, fs = librosa.load(audio, sr=None)
    data = data/max(np.abs(data))
    MFCC = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc, hop_length=int(win_length/2), win_length=win_length, n_fft=win_length, dct_type=n_dct_type)
    delta = librosa.feature.delta(MFCC)
    MFCC = np.append(MFCC, delta, axis=1)
    MFCC_list.append(MFCC)
    records.append(audio)
MFCC_temp = []
max_len = max(np.shape(x)[1] for x in MFCC_list)
for item in MFCC_list:
    temp = []
    n_add = max_len - np.shape(item)[1]
    for coeff in item:
      temp.append(np.append(coeff, np.zeros(n_add)))
    MFCC_temp.append(temp)
    del temp
MFCC_list = MFCC_temp
del MFCC_temp

for item in MFCC_list:
    np.reshape(item, (n_mfcc, max_len, 1))

eval_set_base["Record"] = records
eval_set_base["MFCC"] = MFCC_list
os.chdir(root_dir)

del MFCC_list
del records

# Preprocessing of training database - calculating mel-spectrogram's [with normalization and reshape to proper input shape]
MFCC_list = []
for count, audio in enumerate(audio_train_database["Data"]):
    data = audio/max(np.abs(audio))
    MFCC = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc, hop_length=int(win_length/2), win_length=win_length, n_fft=win_length, dct_type=n_dct_type)
    delta = librosa.feature.delta(MFCC)
    MFCC = np.append(MFCC, delta, axis=1)
    MFCC_list.append(MFCC)
MFCC_temp = []
for item in MFCC_list:
    temp = []
    for coeff in item:
        if np.shape(item)[1] >= max_len:
            temp.append(coeff[0:max_len])
        elif np.shape(item)[1] < max_len:
            n_add = max_len - np.shape(item)[1]
            temp.append(np.append(coeff, np.zeros(n_add)))
    MFCC_temp.append(temp)
    del temp
MFCC_list = MFCC_temp
del MFCC_temp

for item in MFCC_list:
    np.reshape(item, (n_mfcc, max_len, 1))

audio_train_database["MFCC"] = MFCC_list
os.chdir(root_dir)

del MFCC_list

# Display of calculated databases --> just comment if you don't want to see it
print("Training Recordings database : ")
display(audio_train_database)

print("Eval Recordings database : ")
display(eval_set_base)

# Current build of the model --> this one reached best results
model = Sequential()
model.add(Conv2D(32, (6, 6), padding="valid", activation="relu", input_shape=(n_mfcc, max_len, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="valid", strides=2, activation="relu", input_shape=(n_mfcc, max_len, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Reshape((int(1), 480)))
model.add(Bidirectional(LSTM(320)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(320))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(170))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(170))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('softmax'))

# Optimizer initialization
if optimizer == 'Adam':
    adam = Adam()
    model.compile(loss=loss_function, metrics=[metric_type, loss_function], optimizer=adam)
elif optimizer == 'SGD':
    sgd = SGD(learning_rate=0.001)
    model.compile(loss=loss_function, metrics=[metric_type, loss_function], optimizer=sgd)

# Display of model summary
model.summary()

# Mixing the database
audio_train_database = audio_train_database.sample(frac=1)

# Preparing the set for the training
X = np.array(audio_train_database['Mel-Spectrogram'].tolist())
y_true = audio_train_database['Label'].tolist()

label_encoder = LabelEncoder()
y_true = to_categorical(label_encoder.fit_transform(y_true))

# Cross-validation --> with sklearn and kfold - version with 11 splits
acc_per_fold = []
loss_per_fold = []
i = 0
kf = KFold(n_splits=11)

for train_index, test_index in kf.split(X, y_true):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_true[train_index], y_true[test_index]

    # EarlyStopping added to avoid possible over-train
    es = EarlyStopping(monitor='loss', mode='min')
    start = datetime.now()
    history = model.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epoch, validation_data=(X_test, y_test),
                        verbose=add_display)
    duration = datetime.now() - start

    results = model.evaluate(X_test, y_test, verbose=add_display)

    print(
        f'Score for fold {i}: {model.metrics_names[0]} of {results[0]}; {model.metrics_names[1]} of {results[1] * 100}%')
    acc_per_fold.append(results[1] * 100)
    loss_per_fold.append(results[0])
    i += 1

    # Model is saved on every iteration due to problems with RAM on Colab version
    model.save(model_dir)

print('Accuracy: ')
print(np.mean(acc_per_fold))
print('Loss: ')
print(np.mean(loss_per_fold))
del start

# Model save to model_dir
model.save(model_dir)

# Loading model and performing prediction
model = load_model(model_dir)

X = np.array(eval_set_base['Mel-Spectrogram'].tolist())
predict = model.predict(X)

values_from_prediction = []
for item in predict:
    values_from_prediction.append(np.max(item))
class_predictions = np.argmax(predict, axis=-1)
print(class_predictions)

# Save of CSV file with prediction results
rd={"file": eval_set_base['Record'],
        "prediction":class_predictions,
      "values":values_from_prediction}
results_to_csv=pd.DataFrame(data=rd)
print(results_to_csv)
os.chdir(CSV_dir)
results_to_csv.to_csv('results_ia.csv', index=False, header = False, sep = ',')
os.chdir(root_dir)

# Evaluation with use of script prepared by lecturer
os.chdir(CSV_dir)
evaluate('results_ia.csv')
os.chdir(root_dir)
