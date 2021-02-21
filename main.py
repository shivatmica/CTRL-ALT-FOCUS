# Imports and Inits
import pickle
import dlib
import pygame
import numpy as np
import cv2

mid_vals = []
pygame.init()
WIDTH, HEIGHT = 720, 400
color = (255, 255, 255)
color_light = (170, 170, 170)

color_dark = (100, 100, 100)
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CTRL+ALT+FOCUS")

Font = pygame.font.SysFont('Courier New', 35)
Font1 = pygame.font.SysFont('Courier New', 15)
Font2 = pygame.font.SysFont('Courier New', 20)

Text = Font.render('Go to Smile Detctor!', True, color)
text1 = Font.render("CTRL+ALT+FOCUS", True, color)
text2 = Font.render('Go to Smile', True, color)

text3 = Font.render('DETECTOR!', True, color)
text4 = Font.render('Go to Eye', True, color)
text5 = Font.render('DETECTOR!', True, color)

face_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'C:\\Users\\sharv\\PycharmProjects\\OPENCV\\haarascade_eye.xml')


def smile():
    global RED_TIMES, GREEN_TIMES
    # Password for the email and saving the password in a database

    RED_TIMES = 0
    GREEN_TIMES = 0
    count = 0
    filepath = "C:\\Users\\sharv\\password.txt"

    with open(filepath) as file:
        file.seek(0)
        content = file.readlines()

    outfile = open('password.pkl', 'wb')
    pickle.dump(content, outfile)

    outfile.close()
    state = None

    # Function to end emails to anyone

    # .XML files and haarcascades to detect the faces and smile
    count = 0

    webcam = cv2.VideoCapture(0)

    # Show the current frame
    while True:
        # Read the current frame from webcam video stream
        _read, frame = webcam.read()

        # If there's an error, abort
        if not _read:
            break
        # Change to grayscale
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces first

        faces = face_detector.detectMultiScale(frame_grayscale)

        # Detect smiles
        # Run smile detection within each of those faces
        def emails():
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
            message = f"""No. of times you were focused: {GREEN_TIMES // 5}. No. of times you were distracted {RED_TIMES // 5}."""
            message1 = f'Percentage of focus = {round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100))}%'
            text6 = Font1.render(message, True, color_light)

            if round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(-1, 40):
                text7 = Font1.render(message1, True, (255, 0, 0))

            elif round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(39, 70):
                text7 = Font1.render(message1, True, (255, 255, 0))

            elif round(((GREEN_TIMES / (GREEN_TIMES + RED_TIMES)) * 100)) in range(69, 100):
                text7 = Font1.render(message1, True, (0, 255, 0))

            text8 = Font1.render(state, True, color_light)

            SCREEN.blit(text6, (10, 180))
            SCREEN.blit(text7, (10, 210))
            SCREEN.blit(text8, (10, 240))

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)
            face = frame[y:y + h, x:x + w]

            face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            smiles = smile_detector.detectMultiScale(face_grayscale, 1.7, 20)

            if len(smiles) > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                RED_TIMES += 1

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                GREEN_TIMES += 1
        count += 1
        if count == 150:
            break
        cv2.imshow('Smile Detector', frame)

        # Display
        cv2.waitKey(1)
    emails()

def eye():
    count1 = 0

    def format_shape(shape, dtype="int"):
        cords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            cords[i] = (shape.part(i).x, shape.part(i).y)
        return cords

    def masking_the_eye(mask, side):
        p = [shape[i] for i in side]
        p = np.array(p, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, p, 255)
        return mask

    def outlining(thresh, mid, img, right=False):
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
    Moving_count = 0
    still_count = 0
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detect(gray, 1)
        for rect in rects:
            shape = predict(gray, rect)
            shape = format_shape(shape)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            mask = masking_the_eye(mask, left)
            mask = masking_the_eye(mask, right)
            mask = cv2.dilate(mask, kernel, 5)

            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]

            mid = (shape[42][0] + shape[39][0]) // 2
            mid_vals.append(mid)
            for n in range(len(mid_vals)):
                if mid_vals[n] - mid_vals[n - 1] in range(0, 50):
                    still_count += 1
                else:
                    Moving_count += 1

            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)

            thresh = cv2.erode(thresh, None, iterations=2)  # 1
            thresh = cv2.dilate(thresh, None, iterations=4)  # 2
            thresh = cv2.medianBlur(thresh, 3)  # 3

            thresh = cv2.bitwise_not(thresh)
            outlining(thresh[:, 0:mid], mid, img, False)
            outlining(thresh[:, mid:], mid, img, True)

            # for (x, y) in shape[36:48]:
            #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        # show the image with the face detections + facial landmarks
        count1 += 1
        cv2.imshow('eyes', img)
        cv2.imshow("image", thresh)
        cv2.waitKey(1)

        if count1 == 50:
            break

    if still_count > Moving_count:
        text9 = Font2.render('Mostly focused during the time period.', True, (0, 255, 0))
    else:
        text9 = Font2.render('Mostly not focused during the time period', True, (255, 0, 0))

    SCREEN.blit(text9, (30, 300))


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if 0 <= MOUSE[0] <= WIDTH / 2 and 70 <= MOUSE[1] <= 170:
                smile()
            if WIDTH / 2 <= MOUSE[0] <= WIDTH and 70 <= MOUSE[1] <= 170:
                print('eye')
                eye()
    MOUSE = pygame.mouse.get_pos()
    pygame.draw.rect(SCREEN, (255, 0, 0), [0, 70, WIDTH / 2, 100])
    pygame.draw.rect(SCREEN, (0, 255, 0), [WIDTH / 2, 70, WIDTH, 100])

    pygame.draw.line(SCREEN, (255, 255, 255), (0, 280), (WIDTH, 280))
    SCREEN.blit(text1, (25, 25))

    SCREEN.blit(text2, (50, 80))
    SCREEN.blit(text3, (85, 110))

    SCREEN.blit(text4, (430, 80))
    SCREEN.blit(text5, (430, 110))

    pygame.display.update()

cv2.destroyAllWindows()

