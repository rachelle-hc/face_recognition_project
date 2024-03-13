# face1_data.py
#using haarcascades for the static img analysis
import cv2
import face_recognition
import os

def load_known_faces():
    known_face_encodings = []
    known_names = []
    face_cascade = cv2.CascadeClassifier("C:\\Users\\dell\\Desktop\\cv-ip\\cascades\\haarcascade_frontalface_default.xml")

    for image in os.listdir('Faces3'):
        img_path = os.path.join('Faces3', image)
        face_img = cv2.imread(img_path)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Use Haar cascades for face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Assuming only one face is present in each image
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            # Encode the face region using face_recognition library
            face_encoding = face_recognition.face_encodings(face_img, [(y, x+w, y+h, x)])[0]

            known_face_encodings.append(face_encoding)
            known_names.append(image)
        #else:
            #print(f"No face detected in {image}, so deleting")
            #os.remove(img_path)

    return known_face_encodings, known_names
