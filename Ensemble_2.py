from PIL import Image as PILImage  # Rename the Image module to avoid conflict
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
from PIL import ImageOps
import numpy as np
import math
import imutils
import tensorflow as tf
from imutils import face_utils

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import time
noisy_datapoints = 0
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def getShape(image_data):

    # image_path = r"captured_frame.jpg"
    # image = PILImage.open(image_path).convert("RGB")
    image = PILImage.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)).convert("RGB")

    #1)SNN

    # Disable scientific notation for clarity 
    np.set_printoptions(suppress=True)

    # Load the model
    snn_model = tf.keras.models.load_model(r"converted_keras/keras_model.h5", compile=False)

    # Load the labels
    with open(r"converted_keras/labels.txt", "r") as file:
        class_names = file.readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    from PIL import Image

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = snn_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    classes=["Heart","Oblong","Oval","Round","Square"]
    predictions_snn = -1
    for i in range (len(classes)):
        if(classes[i] in class_name):
            predictions_snn = i

    confidence_snn = confidence_score

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
            return 'Heart'
        elif classNo == 1:
            return 'Oblong'
        elif classNo == 2:
            return 'Oval'
        elif classNo == 3:
            return 'Round'
        elif classNo == 4:
            return 'Square'
    img_counter = 0
    # NoFilter = cv2.imread(image_path)
    NoFilter=image_data

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
    # print(f"CLASS: {classIndex} {getCalssName(classIndex)}")
    # print(f"PROBABILITY: {round(probabilityValue * 100, 2)}%")

    # predictions_cnn = classIndex
    # confidence_cnn = probabilityValue

    classes=["Heart","Oblong","Oval","Round","Square"]
    predictions_cnn=-1
    for i in range (len(classes)):
        if(classes[i]==getCalssName(classIndex)):
            predictions_cnn = i
    confidence_cnn = probabilityValue


    #3)KNN
    
    knn_model_file = "KNN/knn_best_model.pkl"
    best_model = joblib.load(knn_model_file)
    def extract_landmark_features():
        global noisy_datapoints

        landmark_features = []

        image = image_data
        image = cv2.resize(image, (500, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        if len(faces) != 1:
            noisy_datapoints += 1
            return None

        for face in faces:
            landmarks = predictor(gray, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                cv2.circle(gray, (x, y), 2, (0, 255, 0), -1)
                landmark_features.append((x, y))

        return landmark_features

    x_predict = []
    facepoints = extract_landmark_features()
    if facepoints is None:
        print("None")
    x_predict.append(facepoints)

    x_predict = np.array(x_predict)

    try:
        n_samples, n_landmarks, n_coordinates = x_predict.shape
        x_predict = x_predict.reshape(n_samples, n_landmarks * n_coordinates)
        y_predicted = best_model.predict(x_predict)
        y_predicted_proba = best_model.predict_proba(x_predict)
        classes=["heart","oblong","oval","round","square"]
        y_confidence=["heart","oblong","oval","round","square"]
        y_confidence = dict(zip(y_confidence, y_predicted_proba[0]))

        # print(y_predicted[0])
        # print(y_confidence[y_predicted[0]])
        predictions_knn=-1
        for i in range (len(y_confidence)):
            if(classes[i]==y_predicted[0]):
                predictions_knn = i
        confidence_knn = y_confidence[y_predicted[0]]
    except ValueError:
        # Handle the exception, for example, by setting default values
        predictions_knn=-1
        confidence_knn=0




    #4)SVM

    def preprocess_SVM(image_data):
        
        def get_lum(image, x, y, w, h, k):
        
            i1 = range(int(-w / 2), int(w / 2))
            j1 = range(0, h)
        
            lumar = np.zeros((len(i1), len(j1)))
            for i in i1:
                for j in j1:
                    if y + k * h < 0 or y + k * h >= image.shape[0]:
                        lumar[i][j] = None
                    else:
                        lum = np.min(np.clip(image[y + k * h, x + i], 0, 255))
                        lumar[i][j] = lum
        
            return np.min(lumar)
        
        
        def q(landmarks, index1, index2):
        
            x1 = landmarks[int(index1)][0]
            y1 = landmarks[int(index1)][1]
            x2 = landmarks[int(index2)][0]
            y2 = landmarks[int(index2)][1]
        
            x_diff = float(x1 - x2)
        
        
            if y1 < y2: y_diff = float(np.absolute(y1 - y2))
            if y1 >= y2:
                y_diff = 0.1
        
            return np.absolute(math.atan(x_diff / y_diff))
        
        
        def d(landmarks, index1, index2):
        
            x1 = landmarks[int(index1)][0]
            y1 = landmarks[int(index1)][1]
            x2 = landmarks[int(index2)][0]
            y2 = landmarks[int(index2)][1]
        
            x_diff = (x1 - x2) ** 2
            y_diff = (y1 - y2) ** 2
        
            dist = math.sqrt(x_diff + y_diff)
        
            return dist  
        
        img = image_data
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        img_f = cv2.resize(img_eq, (500, 600))
        
        faceCascade = cv2.CascadeClassifier("SVM/haarcascade_frontalface_default.xml")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        mn = 14
        faces = faceCascade.detectMultiScale(img_f, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        detector_flag = False
        flag2 = 0
        
        x, y, w, h = 0, 0, 0, 0
        while len(faces) != 1:
            prev_mn = mn
            if len(faces) == 0:
                faces = faceCascade.detectMultiScale(img_f, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                            flags=cv2.CASCADE_SCALE_IMAGE)
                mn -= 1
                if mn == 0 and len(faces) == 0:
                    faces = detector(img_f)
                    detector_flag = True
                    if len(faces) == 0:
                        x = 10
                        y = 10
                        w = img_f.shape[1] - 10
                        h = img_f.shape[0] - 10
                        detector_flag = False
                    break
            else:
                faces = faceCascade.detectMultiScale(img_f, scaleFactor=1.1, minNeighbors=mn, minSize=(100, 100),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
                mn += 1
        
            if prev_mn < mn:
                flag2 = 1
                break
        
        if mn == 0 and detector_flag is True:
            face = faces[flag2]
            x, y = face.left(), face.top()
            w, h = face.right() - x, face.bottom() - y
        elif len(faces) != 0 and mn >= 0:
            face = faces[flag2]
            x, y, w, h = face
        
        cv2.rectangle(img_f, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        dlib_rect = dlib.rectangle(int(0.8 * x), int(0.8 * y), int(x + 1.05*w), int(y + 1.1*h))
        
        detected_landmarks = predictor(img_f, dlib_rect).parts()
        
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
            
        image_copy = img_f.copy()
        
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
        
            cv2.circle(image_copy, pos, 5, color=(255, 0, 0), thickness=2)
        
        
        
        p27 = (landmarks[27][0, 0], landmarks[27][0, 1])
        x = p27[0]
        y = p27[1]
        
        diff = get_lum(img_f, x, y, 8, 2, -1)
        limit = diff - 55
        
        while diff > limit:
            y = int(y - 1)
            diff = get_lum(img_f, x, y, 6, 2, -1)
        
        cv2.circle(image_copy, (x, y), 5, color=(0, 0, 255), thickness=2)
        
        lmark = landmarks.tolist()
        p68 = (x, y)
        lmark.append(p68)
        
        f = []
        
        fwidth = d(lmark, 0, 16)
        fheight = d(lmark, 8, 68)
        f.append(fheight / fwidth)
        
        jwidth = d(lmark, 4, 12)
        f.append(jwidth / fwidth)
        
        hchinmouth = d(lmark, 57, 8)
        f.append(hchinmouth / fwidth)
        ref = q(lmark, 27, 8)
        
        for k in range(0, 17):
            if k != 8:
                theta = q(lmark, k, 8)
                f.append(theta)
        
        for k in range(1, 8):
            dist = d(lmark, k, 16-k)
            f.append(dist/fwidth)
        
        return f

    svm_model=joblib.load("SVM/SVM_model.pickle")

    x_points=np.array(preprocess_SVM(image_data))
    x_points=x_points.reshape(1,-1)
    y_predicted=svm_model.predict(x_points)
    decision_values = svm_model.decision_function(x_points)
    confidence_scores = 1 / (1 + np.exp(-decision_values)) 

    y_confidence=["Heart","Oblong","Oval","Round","Square"]
    y_confidence = dict(zip(y_confidence, confidence_scores[0]))
    # print(y_predicted[0])
    # print(y_confidence[y_predicted[0]])

    classes=["Heart","Oblong","Oval","Round","Square"]
    predictions_svm=-1
    for i in range (len(y_confidence)):
        if(classes[i]==y_predicted[0]):
            predictions_svm = i
    confidence_svm = y_confidence[y_predicted[0]]

    from collections import Counter

    # Combine predictions
    all_predictions = [predictions_snn, predictions_cnn, predictions_knn, predictions_svm]
    print(all_predictions)
    counter = Counter(all_predictions)
    majority_prediction = counter.most_common(1)[0][0]

    # Compare confidence scores in case of a tie
    if counter[majority_prediction] > 1:
        confidence_scores = [confidence_snn, confidence_cnn, confidence_knn, confidence_svm]
        max_confidence = max(confidence_scores)

        # Check if there is a unique maximum confidence score
        if confidence_scores.count(max_confidence) == 1:
            final_prediction = majority_prediction
        else:
            # There is a tie in confidence scores as well, handle as needed
            final_prediction = "Tie in confidence scores"
    else:
        final_prediction = majority_prediction

    print("Final Prediction:", final_prediction)

    from dimmeasure import process_image

    dimensions=process_image("FaceShape_Dataset/heart (1).jpg")
    forehead_width=dimensions[0]
    cheekbone_width=dimensions[1]
    height_eye_to_chin=dimensions[2]

    if final_prediction == 0:
        print("Suitable Frame Shapes: Cat-eye, Round")

        print("Dimensions of the Frame:")
        print(f"Frame Width: {round(1.33*forehead_width,2)} mm")
        print("Lens Height: 40-60 mm")
        print("Bridge Size: 15-25 mm")
        print("Temple Length: 130-150 mm")
        print()
        print("Rounded bases to balance and complement the face")

    elif final_prediction == 1:
        print("Suitable Frame Shapes: Browline")

        print("Dimensions of the Frame:")
        print("Frame Width: 130-150 mm")
        print("Lens Height: 40-55 mm")
        print("Bridge Size: 15-20 mm")
        print("Temple Length: 130-150 mm")
        print()
        print("Browline frames bring more balance to a narrow forehead")

    elif final_prediction == 2:
        print("Suitable Frame Shapes: Cat-eye, Round")

        print("Dimensions of the Frame:")
        print(f"Frame Width: {round(1.25*cheekbone_width,2)} mm")
        print("Lens Height: 40-60 mm")
        print("Bridge Size: 15-25 mm")
        print("Temple Length: 130-150 mm")
        print()
        print("Sharper frames for angular contrast")

    elif final_prediction == 3:
        print("Suitable Frame Shape: Geometric")

        print("Dimensions of the Frame:")
        print("Frame Width: 130-150 mm")
        print("Lens Height: 40-60 mm")
        print("Bridge Size: 15-20 mm")
        print("Temple Length: 130-150 mm")
        print()
        print("Sharper frames for angular contrast")

    elif final_prediction == 4:
        print("Suitable Frame Shape: Round")

        print("Dimensions of the Frame:")
        print("Frame Width: 120-140 mm")
        print("Lens Height: 40-55 mm")
        print("Bridge Size: 15-20 mm")
        print("Temple Length: 130-150 mm")
        print()
        print("Thinner frames that do not overwhelm")

    else:
        pass

    return final_prediction