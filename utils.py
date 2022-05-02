import json
import numpy as np

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def print_report(y_test, y_pred):
    print("Accuracy Score is:\t {0:.3f}".format(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def load_mfcc_data(data_path):

    with open(data_path,"r") as fp:
        data = json.load(fp)

    genres = data['mapping']
    x = np.array(data['mfcc'])
    y = np.array(data["labels"])

    return genres, x, y


def load_multimodal_data(data_path):
   
    with open(data_path,"r") as fp:
        data = json.load(fp)
    
    genres = data['mapping']
    mfcc = np.array(data['mfcc'])
    features = np.array(data['features'])
    y = np.array(data["labels"])
    
    return genres, mfcc, features, y


def plot_history(history):
    
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def test_train_val_split(X, y, test_size=0.3, val_size=0.3, rseed=0):

    # generate test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rseed)

    # generate train/val sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=rseed)

    return X_train, X_val, X_test, y_train, y_val, y_test


def setup_callbacks(model, modelpath):
    saveBest = ModelCheckpoint(filepath=modelpath, verbose=1, save_best_only=True)

    es = EarlyStopping(monitor='val_loss',
                    mode='min',
                    restore_best_weights=True,
                    verbose=1,
                    patience=15)
    es.set_model(model)

    lr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        verbose=1,
                        mode='min',
                        min_delta=0.00001)

    return saveBest, es, lr

def predict_genre(model, song_mfcc, y_true):
    """
    Given a song, predict it's label (using a keras model)
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    song_mfcc = song_mfcc[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(song_mfcc)

    # get index with max value
    y_pred = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y_true, y_pred))


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = keras.models.Model(layers[0].input, x)
    return new_model


def build_multimodal_mlp(INPUT_SHAPE):

    model = keras.Sequential()
    
    # input layer
    model.add(keras.layers.InputLayer(input_shape=INPUT_SHAPE))
    
    # 1st dense layer
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    
    # 2nd dense layer
    model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    
    # 3rd dense layer
    model.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    
    return model


def build_multimodal_cnn(INPUT_SHAPE):

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())


    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    return model


def generate_param_grid(clf):
    if clf == 'knn':
        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 15],
            'leaf_size': [15, 30, 45]
        }
    elif clf == 'svm':
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf'],
        }
    elif clf == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 30, 50, 70]
        }
    elif clf == 'xgb':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 30, 50, 70]
        }

    return param_grid
