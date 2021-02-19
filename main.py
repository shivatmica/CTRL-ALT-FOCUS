import cv2
import numpy as np

class P(object):
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.detect_iris(eye_frame)
        
    @staticmethod
    def image_processing(eye_frame, threshold):
        k = np.ones((3, 3), np.uint8)
        nf = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        nf = cv2.erode(nf, k, iterations=3)
        nf = cv2.threshold(nf, threshold, 255, cv2.THRESH_BINARY)[1]
        return nf
    
    def detect_iris(self, eye_frame):
        self.fi = self.image_processing(eye_frame, self.threshold)
        ct, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ct = ct[-2:]
        cont = sorted(ct, cv2.contourArea)
        try:
            moments = cv2.moments(cont[-2])
            m1, m2, m3 = moments['m10'], moments['m00'], moments['m01']
            self.x = int(m1 / m2)
            self.y = int(m3 / m2)
        except (IndexError, ZeroDivisionError):
            # ZERO Division and Index error catching
            pass

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read, frame = webcam.read()
    
    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_grayscale)
    smiles = smile_detector.detectMultiScale(frame_grayscale, 1.4, 20)

    for (x, y, w, h) in faces:
        the_face = frame[y: y + h, x: x + w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, 1.4, 20)

        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
