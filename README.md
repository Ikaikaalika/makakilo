# makakilo

Here is the complete README.md file in Markdown format:

# Video Analysis Project with YOLOv8 and Facial Recognition

This project utilizes **YOLOv8** for object detection, **DeepFace** for emotion and demographics detection, and **face_recognition** for identifying and remembering individuals in a video feed. The system is designed to process live webcam input or video files, annotate frames with detected objects, emotions, demographics, and recognized faces, and save the processed output.

---

## Features

- **Object Detection**: Detect people in video frames using YOLOv8.
- **Facial Recognition**: Recognize and remember individuals using a database of known faces.
- **Emotion Detection**: Analyze and display emotions of detected faces.
- **Demographics Detection**: Estimate age and gender of individuals.
- **Output Processing**: Save processed videos with annotated information.

---

## Project Structure

```plaintext
video_analysis_project/
├── data/                      # Folder for storing video files
│   ├── sample_video.mp4
├── known_faces/               # Folder for storing known face images
│   ├── person1.jpg
│   ├── person2.jpg
├── models/                    # YOLOv8 model and configs
│   ├── yolov8s.pt             # Pretrained YOLOv8 model
├── outputs/                   # Folder for storing processed outputs
│   ├── processed_video.mp4
├── scripts/                   # Python scripts for the project
│   ├── object_detection.py    # Face detection and analysis logic
│   ├── face_recognition_util.py # Face recognition functionality
│   ├── main.py                # Main entry point for the project
├── requirements.txt           # List of Python dependencies
├── README.md                  # Documentation for the project

Installation

1. Clone the Repository

git clone https://github.com/your-username/video-analysis-project.git
cd video-analysis-project

2. Set Up a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

3. Install Dependencies

pip install -r requirements.txt

Usage

Running the Project

To process a live webcam feed:

python scripts/main.py

To save the processed video:

python scripts/main.py --output outputs/processed_video.mp4

Adding Known Faces
	1.	Place images of known individuals in the known_faces/ directory.
	2.	Name the image files using the person’s name (e.g., person1.jpg becomes “person1”).

Dependencies
	•	ultralytics==8.0.20: YOLOv8 for object detection.
	•	opencv-python>=4.6.0: For video processing.
	•	face_recognition==1.3.0: For face recognition.
	•	deepface: For emotion and demographics detection.
	•	numpy>=1.22,<1.24: Numerical processing.
	•	matplotlib==3.4.3: Visualization tools.
	•	dlib==19.24.0: Dependency for face_recognition.

Troubleshooting
	•	Dependency Conflicts: Ensure you are using Python 3.10 in a virtual environment.
	•	Camera Access Issues: Allow your terminal or IDE access to the webcam in macOS privacy settings.

Contributing
	1.	Fork the repository.
	2.	Create a feature branch (git checkout -b feature-name).
	3.	Commit changes (git commit -m "Description of changes").
	4.	Push to the branch (git push origin feature-name).
	5.	Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.