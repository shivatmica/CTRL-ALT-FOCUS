import cv2
import smtplib
import pickle

FOCUS_POINTS = 0
filepath = "C:\\Users\\sharv\\password.txt"

with open(filepath) as file:
    file.seek(0)
    content = file.readlines()

outfile = open('password.pkl', 'wb')
pickle.dump(content ,outfile)
outfile.close()
state = None

print('Only gmail emails will be accepted.')
cord = input('Enter the email of your coordinator of school: ')
parent = input('Enter the email of your parent: ')
emails = [cord, parent]

def emails():
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("math.pi.study@gmail.com", f"{content}")
    if FOCUS_POINTS == 0:
        state = 'OK'
    elif FOCUS_POINTS > 0:
        state = 'Good'
    else:
        state = 'Bad'
    message = """The number of points your child had throughout the day was {}. 
                 Our rating for him today is {}""".format(FOCUS_POINTS, state)

    for i in emails:
        s.sendmail("math.pi.study@gmail.com", f'{i}', message)



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

    for (x, y, w, h) in faces:
        the_face = frame[y: y + h, x: x + w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, 1.4, 20)
        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            FOCUS_POINTS -= 10
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            FOCUS_POINTS += 10

    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
emails()
