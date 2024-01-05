import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def process_image(image_path):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read the image.")
        return

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Face Detection
    results_detection = face_detection.process(rgb_frame)

    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract face landmarks using Face Mesh
            results_mesh = face_mesh.process(rgb_frame)
            if results_mesh.multi_face_landmarks:
                landmarks = results_mesh.multi_face_landmarks[0]

                # Convert relative coordinates to absolute coordinates
                landmarks_list = [(int(l.x * iw), int(l.y * ih)) for l in landmarks.landmark]

                # Extract specific landmarks (eye center and chin)
                left_eye = landmarks_list[159]  # Index for left eye center
                right_eye = landmarks_list[386]  # Index for right eye center
                chin = landmarks_list[10]  # Index for chin

                # Calculate distances
                forehead_width = calculate_distance(left_eye, right_eye)
                cheekbone_width = calculate_distance(left_eye, right_eye)
                height_eye_to_chin = calculate_distance(left_eye, chin)

                # return or use the calculated distances as needed
                return (forehead_width,cheekbone_width,height_eye_to_chin)
    
if __name__ == "__main__":
    image_path = "FaceShape_Dataset/heart (1).jpg"  # Replace with the path to your image
    process_image(image_path)
