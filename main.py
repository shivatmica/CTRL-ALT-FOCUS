import cv2
import smtplib
import pickle
import numpy as np

RED_TIMES = 0
GREEN_TIMES = 0
filepath = "C:\\Users\\sharv\\password.txt"

with open(filepath) as file:
    file.seek(0)
    content = file.readlines()

outfile = open('password.pkl', 'wb')
pickle.dump(content ,outfile)
outfile.close()
state = None


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_detector = cv2.CascadeClassifier('haarascade_eye.xml')
count = 0
webcam = cv2.VideoCapture(0)
while 1:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_grayscale)
    smiles = smile_detector.detectMultiScale(frame_grayscale)

    for (x, y, w, h) in faces:
        the_face = frame[y: y + h, x: x + w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, 1.4, 20)
        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            RED_TIMES += 1
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            GREEN_TIMES += 1

    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

    count += 1
    if count == 500:
        webcam.release()
        cv2.destroyAllWindows()


print(f'No. of times you were focues: {GREEN_TIMES // 10}')
print(f'No. of times you were distracted {RED_TIMES // 10}')
