import csv
import cv2
import time
from object_detection import YOLOv8Detector
from face_recognition_util import FaceRecognition
from deepface import DeepFace

def analyze_face(face):
    """
    Analyze the face for age, gender, emotion, and race using DeepFace.
    """
    try:
        results = DeepFace.analyze(face, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
        # If results is a list, return the first analysis
        if isinstance(results, list):
            return results[0]
        return results
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None

def flush_buffer_to_csv(log_buffer, csv_path):
    """
    Write buffered log data to the CSV file.
    """
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(log_buffer)
    log_buffer.clear()  # Clear the buffer after writing

def main(output_path=None, location="default_location"):
    # Initialize models
    detector = YOLOv8Detector()
    face_recognizer = FaceRecognition()

    # Ensure the output directory exists
    import os
    os.makedirs("output", exist_ok=True)

    # Generate a unique CSV filename
    unix_time = int(time.time())
    csv_path = f"output/output_data_{location}_{unix_time}.csv"

    # Initialize CSV file
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame", "Unix Time", "Name", 
            "Age (Confidence)", "Gender (Confidence)", "Emotion (Confidence)", 
            "Race Percentages", "Bounding Box"
        ])

    cap = cv2.VideoCapture(0)
    out = None
    frame_count = 0
    log_buffer = []  # Buffer to store log data

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
            break

        frame_count += 1

        # Detect faces using YOLOv8
        results = detector.detect_objects(frame)
        detections = results[0]

        # Detect faces and extract bounding boxes
        faces = detector.detect_faces(frame, detections)

        # Recognize faces and analyze attributes
        for face, bbox in faces:
            x1, y1, x2, y2 = bbox
            face_crop = frame[y1:y2, x1:x2]

            # Analyze face attributes
            analysis = analyze_face(face_crop)
            if analysis:
                # Extract predictions and confidence levels
                age = analysis['age']
                gender = analysis['gender']
                emotion = analysis['dominant_emotion']
                emotion_confidences = analysis['emotion']
                race_percentages = analysis['race']

                # Recognize face
                name = face_recognizer.recognize_and_add_face(face_crop)

                # Prepare data for logging
                unix_time_now = int(time.time())  # Current Unix timestamp
                log_buffer.append([
                    frame_count, unix_time_now, name,  # Use recognized name
                    f"{age}", f"{gender}", f"{emotion} ({emotion_confidences[emotion]:.2f})",
                    race_percentages, bbox
                ])

                # Annotate the frame with bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{name}, {age}, {gender}, {emotion}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # Periodically flush the buffer to CSV
        if frame_count % 15 == 0 and log_buffer:
            flush_buffer_to_csv(log_buffer, csv_path)

        # Display and save the frame
        cv2.imshow('Makakilo - Face Analysis', frame)
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Flush remaining data in the buffer to CSV
    if log_buffer:
        flush_buffer_to_csv(log_buffer, csv_path)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Makakilo analysis.")
    parser.add_argument('--output', type=str, help="Path to save the processed video (optional)")
    parser.add_argument('--location', type=str, default="default_location", help="Location identifier for output files")
    args = parser.parse_args()

    main(output_path=args.output, location=args.location)