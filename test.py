import cv2
import numpy as np
import face_recognition
import os
from threading import Thread

face = cv2.CascadeClassifier("C:\\Users\\dell\\Desktop\\cv-ip\\cascades\\haarcascade_frontalface_default.xml")

known_face_encodings = []
known_names = []

def load_known_faces():
    global known_face_encodings, known_names
    for image in os.listdir('Faces2'):
        img_path = os.path.join('Faces2', image)
        face_img = cv2.imread(img_path)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            face_encoding = face_recognition.face_encodings(face_img, [(y, x+w, y+h, x)])[0]

            known_face_encodings.append(face_encoding)
            known_names.append(image)
        else:
            print(f"No face detected in {image}, so deleting")
            os.remove(img_path)

def recognize_faces(video_capture):
    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_encoding = face_recognition.face_encodings(frame, [face_location])
            matches = face_recognition.compare_faces(known_face_encodings, np.array(face_encoding), tolerance=0.6)
            name = "unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, bottom-6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    load_thread = Thread(target=load_known_faces)
    load_thread.start()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    recognize_faces(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
