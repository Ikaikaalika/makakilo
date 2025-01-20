import cv2
from object_detection import YOLOv8Detector

def main(output_path=None):
    # Step 1: Initialize YOLOv8 Detector
    detector = YOLOv8Detector()

    # Step 2: Access the MacBook webcam
    cap = cv2.VideoCapture(0)  # 0 indicates the default webcam

    # Define video writer if saving output
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Set FPS for output video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Step 3: Perform YOLOv8 object detection
        results = detector.detect_objects(frame)

        # Step 4: Crop faces from detections
        detections = results[0]  # First result corresponds to the current frame
        faces = detector.detect_faces(frame, detections)

        # Step 5: Analyze faces for emotions and demographics
        analyses = detector.analyze_faces(faces)

        # Step 6: Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Overlay demographic and emotion data on the frame
        for idx, analysis in enumerate(analyses):
            text = f"{analysis['dominant_emotion']}, {analysis['age']}, {analysis['gender']}"
            cv2.putText(annotated_frame, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow('YOLOv8 Detection with Emotion and Demographics', annotated_frame)

        # Save the annotated frame if output_path is provided
        if out:
            out.write(annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Video stream ended.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv8 Webcam Detection Tool with Emotion and Demographics')
    parser.add_argument('--output', type=str, help='Path to save the processed video (optional)')
    args = parser.parse_args()

    main(output_path=args.output)