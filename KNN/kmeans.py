import dlib
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

scaler = StandardScaler()

kmeans_model_file = "knn/kmeans_model.pkl"
landmarks_file = "knn/landmarks.pkl"
labels_file = "knn/labels.pkl"

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

if not os.path.exists(kmeans_model_file):
    if not os.path.exists(landmarks_file) or not os.path.exists(labels_file):
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

        landmark_features = np.array(landmark_features)
        landmark_labels = np.array(landmark_labels)

        n_samples, n_landmarks, n_coordinates = landmark_features.shape
        landmark_features = landmark_features.reshape(n_samples, n_landmarks * n_coordinates)

        joblib.dump(landmark_features,landmarks_file)
        joblib.dump(landmark_labels,labels_file)
    else:
        landmark_features = joblib.load(landmarks_file)
        landmark_labels = joblib.load(labels_file)

    x_train, x_test, y_train, y_test = train_test_split(landmark_features, landmark_labels, test_size=0.2,
                                                        random_state=42)

    num_clusters = len(np.unique(y_train))

    param_grid = {
    'n_clusters': [2, 3, 4, 5],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [100, 200, 300]
    }

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    kmeans_model = KMeans()
    grid_search = GridSearchCV(kmeans_model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train)
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.transform(x_test)

    cluster_distance = []
    for distance, predict in zip(y_pred_proba, y_pred):
        confidence_scores = [1 / (1 + dist) for dist in distance]
        cluster_distance.append(confidence_scores)
    cluster_distance_array = np.array(cluster_distance)

    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(y_pred)

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
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], cluster_distance_array[:, i])
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

    joblib.dump(best_model, kmeans_model_file)
else:
    kmeans_model = joblib.load(kmeans_model_file)

# x_image_paths = []
# test_path = "knn/FaceShape_testing"
# for root, dirs, files in os.walk(test_path):
#     for file in files:
#         if file.endswith(".jpg") or file.endswith(".jpeg"):
#             x_image_paths.append(os.path.join(root, file))

# x_predict = []
# y_labels = []
# for path in x_image_paths:
#     facepoints = extract_landmark_features(path)
#     if facepoints is None:
#         continue
#     x_predict.append(facepoints)
#     y_labels.append(path.split("\\")[1].split(".")[0])

print("start")
x_predict=[]
y_labels=[]
image=r"C:\Users\Bhargav\Downloads\glass shape recommender\FaceShape_Training_Set\oblong (511).jpg"
facepoints=extract_landmark_features(image)
print("landmarks")
x_predict.append(facepoints)
y_labels.append(image.split("\\")[1].split(".")[0])
x_predict = np.array(x_predict)
n_samples, n_landmarks, n_coordinates = x_predict.shape
x_predict = x_predict.reshape(n_samples, n_landmarks * n_coordinates)
print("reshaped")
cluster_labels_predict = kmeans_model.predict(x_predict)
centroid_distances = kmeans_model.transform(x_predict)
print("done")
cluster_to_shape = {
    1: 'Heart',
    2: 'Round',
    3: 'Square',
    4: 'Oval',
    5: 'Oblong',
}

for label, predict, distances in zip(y_labels, cluster_labels_predict, centroid_distances):
    facial_shape = cluster_to_shape.get(predict, 'Unknown')
    confidence = 1 / (1 + distances[predict])
    print(f"{label}: {facial_shape} (Confidence: {confidence:.2f})")