from PIL import Image
import dlib
import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import math
import imutils
import tensorflow as tf
from imutils import face_utils

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


#2)CN

cnn_model = tf.keras.models.load_model('CNN/faceshape_model.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Face Shape Heart'
    elif classNo == 1:
        return 'Face Shape Oblong'
    elif classNo == 2:
        return 'Face Shape Oval'
    elif classNo == 3:
        return 'Face Shape Round'
    elif classNo == 4:
        return 'Face Shape Square'

def getShape(imgOrignalin,NoFilterin):
    imgOrignal = imgOrignalin
    NoFilter = NoFilterin

    img = np.asarray(NoFilter)
    img = cv2.resize(img, (32, 32))

    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = cnn_model.predict(img)
    # print(predictions)
    classIndex = np.argmax(predictions, axis=-1)

    # print("Class Index", classIndex)
    probabilityValue = np.amax(predictions)

    # print(probabilityValue)
    print(f"CLASS: {classIndex} {getCalssName(classIndex)}")
    print(f"PROBABILITY: {round(probabilityValue * 100, 2)}%")

    return classIndex

