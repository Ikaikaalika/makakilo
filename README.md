Here’s the properly formatted Markdown version of your README content:

# Makakilo - Real-Time Face Analysis and Recognition

Makakilo is a Python-based real-time face detection, analysis, and recognition system. The name “Makakilo” comes from the Hawaiian words “maka” (eye) and “kilo” (observe), embodying the project’s goal of observing and analyzing human faces dynamically and effectively.

## Features
1. **Real-Time Face Detection**:
   - Utilizes YOLOv8 for high-speed and accurate object detection.
2. **Face Analysis**:
   - Extracts attributes such as **age**, **gender**, **emotion**, and **racial composition percentages** using DeepFace.
3. **Face Recognition**:
   - Assigns unique IDs to individuals, remembers them across sessions, and stores their cropped face images.
4. **Data Logging**:
   - Logs real-time data, including frame count, timestamps, recognized IDs, attributes, and bounding boxes, into a CSV file.
5. **Visual Feedback**:
   - Displays bounding boxes and overlays detected attributes directly on the video feed.

---

## Setup and Installation

### 1. Prerequisites
- Python 3.10 or later
- An M1/M2 MacBook or a machine with similar capabilities
- A webcam or video input device

### 2. Clone the Repository
```bash
git clone https://github.com/your-repo/makakilo.git
cd makakilo

3. Create and Activate Virtual Environment

Using Conda:

conda create --name makakilo python=3.10 -y
conda activate makakilo

Using venv:

python3 -m venv makakilo_env
source makakilo_env/bin/activate

4. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

Usage

Run the Application

Start the face detection and analysis system:

python scripts/main.py --output output/video_output.mp4 --location "your_location"

Arguments
	•	--output: Path to save the processed video (optional).
	•	--location: A location identifier included in the CSV filename (default: "default_location").

Directory Structure

makakilo/
├── scripts/
│   ├── main.py                  # Main script to run the project
│   ├── object_detection.py      # YOLOv8 detection implementation
│   ├── face_recognition_util.py # Face recognition and ID management
├── known_faces/                 # Directory to store known faces
├── output/                      # Directory for output CSV and videos
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation

Outputs
	1.	Video File:
	•	Annotated video with bounding boxes and attributes for each detected face.
	2.	CSV File:
	•	Logs the following data for every detected face:
	•	Frame Number
	•	Unix Time
	•	Unique ID
	•	Age (with confidence)
	•	Gender (with confidence)
	•	Emotion (with confidence)
	•	Race Percentages
	•	Bounding Box Coordinates

How It Works
	1.	Detection:
	•	YOLOv8 detects faces and provides bounding boxes.
	2.	Analysis:
	•	DeepFace analyzes the cropped face for attributes like age, gender, emotion, and racial composition.
	3.	Recognition:
	•	If the face is unrecognized, it is assigned a unique ID, stored, and remembered in future sessions.
	4.	Logging:
	•	Data is periodically saved to a CSV file for further analysis.
	5.	Visualization:
	•	Bounding boxes and attributes are drawn on the video feed.

Hawaiian Connection

The name “Makakilo” honors the Hawaiian language and culture, translating to “observing eyes.” This project symbolizes the act of intelligent observation and data collection, aligning with its purpose of advanced face recognition and analysis.

Future Enhancements
	•	Integration with cloud storage for real-time data synchronization.
	•	Advanced emotion recognition using neural networks.
	•	Support for multi-camera inputs.
	•	Optimization for faster processing.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

You can copy-paste this content into your `README.md` file to ensure proper Markdown formatting. Let me know if you need further assistance!