from ultralytics import YOLO
import cv2

class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8n.pt'):
        # Load YOLOv8 model
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        # Perform YOLOv8 inference
        results = self.model.predict(source=frame, show=False, save=False)
        return results

    def detect_faces(self, frame, detections):
        """
        Extract faces from YOLOv8 detections for further analysis.
        :param frame: Input frame (image).
        :param detections: YOLOv8 detection results.
        :return: List of tuples containing face crops and bounding boxes.
        """
        faces = []
        for box in detections.boxes:  # Iterate through detected boxes
            cls = int(box.cls[0])  # Class ID (0 = person in COCO dataset)
            if cls == 0:  # Check if detection is a person
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                face = frame[y1:y2, x1:x2]  # Crop the face
                print(f"Face detected: {face}, BBox: {x1, y1, x2, y2}")
                faces.append((face, (x1, y1, x2, y2)))  # Add the face and its bounding box

        return faces