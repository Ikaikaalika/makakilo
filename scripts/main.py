import cv2
from object_detection import YOLOv8Detector
from face_recognition_util import FaceRecognition

def main(output_path=None):
    # Initialize YOLOv8 Detector and Face Recognition
    detector = YOLOv8Detector()
    face_recognizer = FaceRecognition()

    # Open webcam
    cap = cv2.VideoCapture(0)
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Step 1: Perform YOLOv8 object detection
        results = detector.detect_objects(frame)

        # Step 2: Crop faces from detections
        detections = results[0]
        face_locations = detector.detect_faces(frame, detections)

        # Step 3: Recognize faces
        recognized_faces = face_recognizer.recognize_faces(frame, face_locations)

        # Step 4: Annotate frame with recognition data
        for name, location in recognized_faces:
            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display and save the frame
        cv2.imshow('YOLOv8 with Face Recognition', frame)
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Video stream ended.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv8 Webcam Detection with Face Recognition')
    parser.add_argument('--output', type=str, help='Path to save the processed video (optional)')
    args = parser.parse_args()

    main(output_path=args.output)