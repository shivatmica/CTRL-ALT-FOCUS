# Imports and Inits
import pickle
import dlib
import numpy as np
import cv2

face_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarascade_eye.xml')

def smile():
    # Password for the email and saving the password in a database
    RED_TIMES = 0
    GREEN_TIMES = 0

    filepath = "C:\\Users\\sharv\\password.txt"

    with open(filepath) as file:
        file.seek(0)
        content = file.readlines()

    outfile = open('password.pkl', 'wb')
    pickle.dump(content, outfile)

    outfile.close()
    state = None

    print('Only google email accepted.')
    parent = input("Enter your parent's number: ")
    password = content[0]


    # Function to end emails to anyone
    def emails():
        # enter all the details
        # get app_key and app_secret by registering
        # a app on sinchSMS

        # Determining the state of the child's focus
        if round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(50, 76):
            state = "Your child was quite focused during this time."

        elif round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(75, 101):
            state = "Your child was very focused during this time."

        elif round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(25, 51):
            state = "Your child was not very focused during this time."

        elif round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(0, 26):
            state = "Your child was not at all focused during this time."

        # Making the content of the message

        # enter the message to be sent
        message = f"""No. of times you were focused: {GREEN_TIMES // 5}.  
                      No. of times you were distracted {RED_TIMES // 5}. 
                      Percentage of focus = {round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100))}%. 
                      {state}"""

        print(message)

    # .XML files and haarcascades to detect the faces and smile
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

            # Detecting the smiles and eyes
            smiles = smile_detector.detectMultiScale(face_grayscale, 1.8, 20)
            eyes = eye_detector.detectMultiScale(face_grayscale, 1.8, 20)
            # Making the green or red rectangle according to the smile
            if len(smiles) > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                RED_TIMES += 1

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                GREEN_TIMES += 1

        count += 1
        if count == 100:
            emails()
            webcam.release()
            cv2.destroyAllWindows()

def eye():
    def shape_to_np(shape, dtype="int"):
        cords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            cords[i] = (shape.part(i).x, shape.part(i).y)
        return cords

    def eye_on_mask(mask, side):
        p = [shape[i] for i in side]
        p = np.array(p, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, p, 255)
        return mask

    def contouring(thresh, mid, img, right=False):
        _, contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
            if right:
                x += mid
            cv2.circle(img, (x, y), 4, (0, 0, 255), 2)
        except:
            pass

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    thresh = img.copy()

    cv2.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    def nothing(x):
        pass

    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = eye_on_mask(mask, left)
            mask = eye_on_mask(mask, right)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)  # 1
            thresh = cv2.dilate(thresh, None, iterations=4)  # 2
            thresh = cv2.medianBlur(thresh, 3)  # 3
            thresh = cv2.bitwise_not(thresh)
            contouring(thresh[:, 0:mid], mid, img, False)
            contouring(thresh[:, mid:], mid, img, True)
            # for (x, y) in shape[36:48]:
            #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        # show the image with the face detections + facial landmarks
        cv2.imshow('eyes', img)
        cv2.imshow("image", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

eye()
cv2.destroyAllWindows()
