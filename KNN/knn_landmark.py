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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

scaler = StandardScaler()
label_encoder = LabelEncoder()

knn_model_file = "knn\knn_best_model.pkl"

noisy_datapoints = 0

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def extract_landmark_features(image_path):
    global noisy_datapoints

    landmark_features = []

    image = cv2.imread(image_path)
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


if not os.path.exists(knn_model_file):
    data_path = "FaceShape_Dataset"
    x_image_paths = []
    y_labels = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                label = file.split(" ")[0]
                x_image_paths.append(os.path.join(root, file))
                y_labels.append(label)

    landmark_features = []
    landmark_labels = []
    for image_path, label in zip(x_image_paths, y_labels):
        facepoints = extract_landmark_features(image_path)
        if facepoints is None:
            continue
        landmark_features.append(facepoints)
        landmark_labels.append(label)

    print(noisy_datapoints)

    landmark_features = np.array(landmark_features)
    landmark_labels = np.array(landmark_labels)

    n_samples, n_landmarks, n_coordinates = landmark_features.shape
    landmark_features = landmark_features.reshape(n_samples, n_landmarks * n_coordinates)

    x_train, x_test, y_train, y_test = train_test_split(landmark_features, landmark_labels, test_size=0.2,
                                                        random_state=42)

    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", best_params)
    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)

    joblib.dump(best_model, knn_model_file)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    classification = classification_report(y_test,y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    class_colors = {'heart': 'red', 'round': 'blue', 'square': 'green', 'oval': 'yellow', 'oblong': 'orange'}
    for feature, label in zip(landmark_features, landmark_labels):
        x_coordinates = feature[::2]
        y_coordinates = feature[1::2]
        color = class_colors.get(label, 'black')
        plt.scatter(x_coordinates, y_coordinates, color=color, s=1)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of Landmark Features with Class Labels')
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    for class_label, color in class_colors.items():
        plt.scatter([], [], color=color, label=class_label)
    plt.legend()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(np.concatenate((y_test, y_pred)))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_bin.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    colors = ['blue', 'red', 'green', 'yellow', 'orange']
    for i, target, color in zip(range(n_classes), classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} (AUC = {1:0.2f})'.format(target, roc_auc[i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
else:
    best_model = joblib.load(knn_model_file)

x_image_paths = []
test_path = "KNN/FaceShape_testing"
for root, dirs, files in os.walk(test_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            label = file.split(" ")[0]
            x_image_paths.append(os.path.join(root, file))

x_predict = []
y_labels = []
for path in x_image_paths:
    facepoints = extract_landmark_features(path)
    if facepoints is None:
        continue
    x_predict.append(facepoints)
    y_labels.append(path.split("\\")[1].split(".")[0])

x_predict = np.array(x_predict)
n_samples, n_landmarks, n_coordinates = x_predict.shape
x_predict = x_predict.reshape(n_samples, n_landmarks * n_coordinates)

x_predict = np.array(x_predict)
y_predicted = best_model.predict(x_predict)
y_predicted_proba = best_model.predict_proba(x_predict)

for label, predict, prob in zip(y_labels, y_predicted, y_predicted_proba):
    print(f"{label}: {predict} (Confidence: {prob.max():.2f})")