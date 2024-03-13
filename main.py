import face_recognition
import cv2
import numpy as np
import os
from face1_data import load_known_faces
#from face2_data import load_known_faces
#from face3_data import load_known_faces



# Load known faces and their encodings
known_face_encodings, known_names = load_known_faces()

# Function to recognize faces
def recognize_faces(video_capture, known_face_encodings, known_names):
    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_encoding = np.array(face_recognition.face_encodings(frame, [face_location]))
            matches = face_recognition.compare_faces(known_face_encodings, np.array(face_encoding), tolerance=0.6)
            name = "unknown"
            #euclidean distance - compare fn
            #compare fn distance measure, so lower is more same

            first_match_index = -1
            first_match_distance = float('inf')

            for idx, match in enumerate(matches):
                if match:
                    first_match_index = idx
                    first_match_distance = face_recognition.face_distance([known_face_encodings[idx]], face_encoding)[0]
                    break

            # Iterate over the remaining matches to find a closer match
            for idx in range(first_match_index + 1, len(matches)):
                if matches[idx]:
                    distance = face_recognition.face_distance([known_face_encodings[idx]], face_encoding)[0]
                    if distance < first_match_distance:
                        first_match_index = idx
                        first_match_distance = distance

            # Assign the name based on the best match found
            if first_match_index != -1:
                name = known_names[first_match_index]



            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Load video capture
cap = cv2.VideoCapture(0)

# Call recognize_faces function
recognize_faces(cap, known_face_encodings, known_names)
