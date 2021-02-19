import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_detector = cv2.CascadeClassifier('haarascade_eye.xml')

webcam = cv2.VideoCapture(0)
while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_grayscale)
    smiles = smile_detector.detectMultiScale(frame_grayscale)
    eyes = eyes_detector.detectMultiScale(frame_grayscale)

    for (x, y, w, h) in faces:
        the_face = frame[y: y + h, x: x + w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, 1.4, 20)
        eyes = eyes_detector.detectMultiScale(face_grayscale, 1.4, 20)
        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
