from ultralytics import YOLO
import cv2

class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8s.pt'):
        # Load YOLOv8 model
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        # Perform YOLOv8 inference
        results = self.model.predict(source=frame, show=False, save=False)
        return results

    def detect_faces(self, frame, detections):
        """
        Extract faces from YOLOv8 detections for further analysis.
        """
        faces = []
        for detection in detections.boxes:  # Iterate through detected boxes
            x1, y1, x2, y2 = detection.xyxy[0]  # Bounding box coordinates
            cls = int(detection.cls[0])  # Class ID (0 = person in COCO)
            
            if cls == 0:  # Check if detection is a person
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                faces.append(face)

        return faces

    def analyze_faces(self, faces):
        """
        Placeholder for analyzing cropped faces (emotion, demographics).
        Can integrate with DeepFace or custom models.
        """
        return faces