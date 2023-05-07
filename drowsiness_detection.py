import cv2
import dlib
import math
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
from datetime import datetime
from imutils import face_utils

video = cv2.VideoCapture(0)

alertSound = "alert.wav"
pygame.init()
sound = pygame.mixer.Sound(alertSound)

if not os.path.exists('records/'):
    os.makedirs('records/')


face_detector = dlib.get_frontal_face_detector() #finds faces from a given image

#face landmarks detecting model
face_landmarks_detector = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        print("Both points must have the same dimension")
   
    squared_distance = sum((p - q) ** 2 for p, q in zip(point1, point2))
    return math.sqrt(squared_distance)

def eyeAspectRatio(eye):
    a = euclidean_distance(eye[1], eye[5])
    b = euclidean_distance(eye[2], eye[4])
    c = euclidean_distance(eye[0], eye[3])
    return  round((a+b) / (2*c), 2)

def mouthAspectRatio(mouth):
    a = euclidean_distance(mouth[2], mouth[10])
    b = euclidean_distance(mouth[4], mouth[8])
    c = euclidean_distance(mouth[0], mouth[6])
    return round((a+b) / (2*c), 2)

def saveImage(frame):
    now = datetime.today().now();
    nameString = now.strftime("%Y-%m-%d-%H-%M-%S")
    savePath = 'records/'+nameString+'.jpeg'
    cv2.imwrite(savePath, frame)


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = (48, 68)

frameFlag = 0
earThresh = 0.25
marThresh = 0.75
frameCheck = 15


while True:
   ret , frame = video.read()  
   # ret: A boolean value that indicates whether the method was 
   # successful in capturing a new frame

   if ret == False:
    print("cannot get frame, exiting")
    break
   else:
    # convert frame to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # pass the grayscale image to the face_detector, it will
    # return all the faces that are in the image. 
    faces = face_detector(gray)

    for face in faces:
        face_landmarks = face_landmarks_detector(gray, face)
        
        shape = face_utils.shape_to_np(face_landmarks)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye);

        ear = (leftEAR + rightEAR) / 2

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        imageSaved = False

        if ear < earThresh:
            frameFlag += 1
            print(frameFlag)
            if frameFlag >= frameCheck:
                print("Drowsy!!!")
                if not imageSaved:
                    saveImage(frame)
                    imageSaved = True
                if not pygame.mixer.get_busy():
                    sound.play()
        else:
            sound.stop()
            frameFlag = 0


        mouth = shape[mStart:mEnd]
        mar = mouthAspectRatio(mouth)

        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        print('mar', mar)
        if mar > marThresh:
            print("Yawning")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
