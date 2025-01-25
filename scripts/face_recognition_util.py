import face_recognition
import os
import cv2
import time

class FaceRecognition:
    def __init__(self, known_faces_path='known_faces/'):
        self.known_encodings = []
        self.known_names = []
        self.known_faces_path = known_faces_path

        # Ensure the known faces directory exists
        os.makedirs(known_faces_path, exist_ok=True)

        # Load known faces
        for image_path in os.listdir(known_faces_path):
            name = os.path.splitext(image_path)[0]
            image = face_recognition.load_image_file(os.path.join(known_faces_path, image_path))
            encoding = face_recognition.face_encodings(image)[0]

            self.known_encodings.append(encoding)
            self.known_names.append(name)

    def recognize_faces(self, frame, face_locations):
        """
        Recognize faces in the current frame.
        :param frame: Current video frame
        :param face_locations: List of face locations detected
        :return: List of recognized names and their locations
        """
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        recognized_faces = []

        for face_encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
            else:
                # Save the new face as an image
                self.save_new_face(frame, location)

            recognized_faces.append((name, location))

        return recognized_faces

    def save_new_face(self, frame, location):
        """
        Save a new face to the known_faces folder.
        :param frame: Current video frame
        :param location: Bounding box of the face (top, right, bottom, left)
        """
        y1, x2, y2, x1 = location  # Ensure proper format
        print(f"Saving face with bounding box: {y1, x2, y2, x1}")  # Debug print
        face_image = frame[y1:y2, x1:x2]
        if face_image.size == 0:  # Check if the cropped face is valid
            print("Error: Cropped face has zero size!")
            return

        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for saving
        timestamp = int(time.time())
        face_filename = os.path.join(self.known_faces_path, f"unknown_{timestamp}.jpg")
        cv2.imwrite(face_filename, face_image_rgb)
        print(f"Saved new face: {face_filename}")
        
    def recognize_and_add_face(self, face):
        """
        Recognize a face and add it to the known faces directory if it's new.
        :param face: Cropped face image.
        :return: Name of the recognized person or "Unknown".
        """
        # Encode the face
        face_encoding = face_recognition.face_encodings(face)
        if not face_encoding:  # Skip if no encoding is found
            return "Unknown"
        face_encoding = face_encoding[0]

        # Compare with known encodings
        matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            # Recognize the first match
            first_match_index = matches.index(True)
            name = self.known_names[first_match_index]
        else:
            # Save the new face
            timestamp = int(time.time())
            face_filename = os.path.join(self.known_faces_path, f"unknown_{timestamp}.jpg")
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            cv2.imwrite(face_filename, face_rgb)
            print(f"New face saved: {face_filename}")

            # Update known faces
            self.known_encodings.append(face_encoding)
            self.known_names.append(f"unknown_{timestamp}")

        return name