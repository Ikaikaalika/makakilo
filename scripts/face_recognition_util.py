import face_recognition
import os

class FaceRecognition:
    def __init__(self, known_faces_path='known_faces/'):
        self.known_encodings = []
        self.known_names = []

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

            recognized_faces.append((name, location))

        return recognized_faces