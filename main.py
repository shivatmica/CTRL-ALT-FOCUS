# Imports and Inits
import pickle
import time

import cv2
from sinchsms import SinchSMS

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
face_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarcascade_frontalface_default.xml')

smile_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarcascade_smile.xml')

eye_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarascade_eye.xml')
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

        # Making the green or red rectangle according to the smile
        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            RED_TIMES += 1

        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            GREEN_TIMES += 1

    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

    count += 1
    if count == 100:
        emails()
        webcam.release()
        cv2.destroyAllWindows()
