from keras.initializers import glorot_normal
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
import numpy.random as rng
from keras import backend as K
import cv2


def W_init(shape, dtype=None):
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name="dtype")


def b_init(shape, dtype=None):
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name="dtype")


input_shape = (96, 96, 1)

model = Sequential()
model.add(Conv2D(32, (4, 4), activation='relu', input_shape=input_shape, kernel_initializer=glorot_normal(seed=None),
                 kernel_regularizer=l2(2e-4)))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(
    Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_normal(seed=None), kernel_regularizer=l2(2e-4)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(
    Conv2D(128, (2, 2), activation='relu', kernel_initializer=glorot_normal(seed=None), kernel_regularizer=l2(2e-4)))
model.add(MaxPooling2D())
# model.add(Dropout(0.3))
model.add(
    Conv2D(256, (1, 1), activation='relu', kernel_initializer=glorot_normal(seed=None), kernel_regularizer=l2(2e-4)))
model.add(MaxPooling2D())
# model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1000, activation="relu", kernel_regularizer=l2(1e-3), kernel_initializer=glorot_normal(seed=None),
                bias_initializer=b_init))
# model.add(Dropout(0.5))
model.add(Dense(1000, activation="relu", kernel_regularizer=l2(1e-3), kernel_initializer=glorot_normal(seed=None),
                bias_initializer=b_init))
# model.add(Dropout(0.6))
model.add(Dense(30, activation=None, kernel_regularizer=l2(1e-3), kernel_initializer=glorot_normal(seed=None),
                bias_initializer=b_init))

model.load_weights('face_landmarks.h5')


def predict_face_landmarks(image_path):
    # Load the input image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the input image to the same size used during training (96x96)
    input_image = cv2.resize(input_image, (96, 96))

    # Normalize the input image
    input_image = input_image / 255.0

    # Reshape the input image to match the model's input shape
    input_image = input_image.reshape(1,96,96, 1)

    # Make predictions on the input image
    predicted_landmarks = model.predict(input_image)

    return predicted_landmarks


# Specify the path to the new input image
image_path = "portrait-of-a-beautiful-girl-with-blood-on-her-face-3d-rendering-ai-generative-image-free-photo.jpg"

# Call the function to predict face landmarks
predicted_landmarks = predict_face_landmarks(image_path)

# Load the input image
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

input_image = cv2.resize(input_image, (96, 96))

input_image = input_image / 255.0

for i in range(0, 30, 2):
    x, y = int(predicted_landmarks[0, i]), int(predicted_landmarks[0, i + 1])
    cv2.circle(input_image, (x, y), 3, (0, 0, 255), -1)

cv2.namedWindow("Input Image with Predicted Landmarks", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Input Image with Predicted Landmarks", 600, 600)

cv2.imshow("Input Image with Predicted Landmarks", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()