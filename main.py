import cv2
import numpy as np
import face_recognition
import os


face=cv2.CascadeClassifier("C:\\Users\\dell\\Desktop\\cv-ip\\cascades\\haarcascade_frontalface_default.xml")

def detector(img): #live face detection
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height,width = img.shape[:2] #extracting values from tuples
    min_size = (int(width * 0.1), int(height * 0.1)) #tuple #experiment values 0.1 etc
    faces=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=min_size) #detectmultiscale is from haarcascades library
    #scalefactor,neighbours
    #we are binding faces to the haarcascade parameters so that we can use the haarcascade paramteres for detecting the face hence face detection function, like is face there or not
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,125),3)
        return img

def load_known_faces(): #dataset face detection, not live
    known_face_encodings = []
    known_names = []
    for image in os.listdir('Faces2'):

        face_image = face_recognition.load_image_file(f"Faces2/{image}")
        '''
        face_recognition.load_image_file() is a function from the face_recognition library, and it's used to load an image from a file into a format that can be processed by the face_recognition library's functions.
        '''
        face_encoding = face_recognition.face_encodings(face_image)
        '''
        face_recognition.face_encodings() is then used to generate face encodings from the loaded image. This function extracts facial features and encodes them into a numerical representation that can be compared with other face encodings to recognize faces.
        from face recognition library
        '''

        if face_encoding:
            face_encoding = face_encoding[0]
            known_face_encodings.append(face_encoding)
            known_names.append(image)
        else:
            print(f"no face detected in {image} so deleting")
            os.remove(f"Faces2/{image}")


    return known_face_encodings,known_names

#function to recognize faces now

def recognize_faces(video_capture,known_face_encodings,known_names):

    while True:
        ret,frame=video_capture.read()
        face_locations=face_recognition.face_locations(frame)

        for face_location in face_locations:
            top,right,bottom,left=face_location
            face_encoding=face_recognition.face_encodings(frame, [face_location])
            matches = face_recognition.compare_faces(known_face_encodings,np.array(face_encoding),tolerance=0.6)
            name="unknown"
            print(face_encoding)
            #figure our compare_Faces what matehmatical thing it is doing
            #like if diff index , if better value is given in the last couple of imgas, but it first picks the first index of true statement,error right

            if True in matches:
                first_match_index=matches.index(True)
                name=known_names[first_match_index]


            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
            font=cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame,name,(left+6,bottom-6),font,0.5,(255,255,255),1)
            #print("recognized name:",name)

        cv2.imshow("video",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

cap=cv2.VideoCapture(0) #caapdshow for solving warning in new update of spyder

#main function

known_face_encodings,known_names = load_known_faces()
recognize_faces(cap,known_face_encodings,known_names)
cap.release()
cv2.destroyAllWindows()
