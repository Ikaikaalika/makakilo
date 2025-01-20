from ultralytics import YOLO
import cv2
from deepface import DeepFace

class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8m.pt'):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        # Perform YOLOv8 inference
        results = self.model.predict(source=frame, show=False, save=False)
        return results

    def detect_faces(self, frame, detections):
        """
        Crop faces from the detected bounding boxes for emotion and demographics analysis.
        """
        faces = []
        for detection in detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2, conf, cls = detection.boxes.xyxy[0].numpy()
            if cls == 0:  # Assuming '0' is the class ID for a person
                # Crop the face region
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                faces.append(face)
        return faces

    def analyze_faces(self, faces):
        """
        Perform emotion and demographics analysis on cropped faces.
        """
        analyses = []
        for face in faces:
            try:
                analysis = DeepFace.analyze(
                    face, actions=['age', 'gender', 'emotion'], enforce_detection=False
                )
                analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing face: {e}")
        return analyses