import cv2
import dlib
import os
import shutil
import math
from datetime import datetime
from imutils import face_utils
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

face_detector = dlib.get_frontal_face_detector() #finds faces from a given image

#face landmarks detecting model
face_landmarks_detector = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def euclideanDistance(point1, point2):
    if len(point1) != len(point2):
        print("Both points must have the same dimension")
   
    squared_distance = sum((p - q) ** 2 for p, q in zip(point1, point2))
    return math.sqrt(squared_distance)

def eyeAspectRatio(eye):
    a = euclideanDistance(eye[1], eye[5])
    b = euclideanDistance(eye[2], eye[4])
    c = euclideanDistance(eye[0], eye[3])
    return   (a+b) / (2*c)

def mouthAspectRatio(mouth):
    a = euclideanDistance(mouth[2], mouth[10])
    b = euclideanDistance(mouth[4], mouth[8])
    c = euclideanDistance(mouth[0], mouth[6])
    return round((a+b) / (2*c), 2)

def saveImage(frame):
    now = datetime.today().now()
    nameString = now.strftime("%Y-%m-%d-%H-%M-%S")
    savePath = 'static/records/'+nameString+'.jpeg'
    cv2.imwrite(savePath, frame)
    return nameString+'.jpeg'

def removeDuplicates(data):
    uniqueData = {};

    for item in data:
        time = item[1];
        if time in uniqueData:
            continue
        uniqueData[time] = item

    return list(uniqueData.values())

def formatTime(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = (49, 68)


earThresh = 0.25
marThresh = 0.80
frameCheck = 20


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('drowsiness.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        return redirect(url_for('detect_drowsiness', filename=filename))


@app.route('/detect_drowsiness/<string:filename>')
def detect_drowsiness(filename):
    file_path = os.path.join('static/uploads', filename)
    video = cv2.VideoCapture(file_path)
    records_dir = 'static/records'
    shutil.rmtree(records_dir, ignore_errors=True)
    os.makedirs(records_dir)

    frameFlag = 0
    image_filenames = [] # list to store image filenames
    duration = 0 # to store the duration of the video

    while True:
        ret, frame = video.read()

        if ret == False:
            print("cannot get frame, exiting")
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            for face in faces:
                face_landmarks = face_landmarks_detector(gray, face)
                
                shape = face_utils.shape_to_np(face_landmarks)
                
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eyeAspectRatio(leftEye)
                rightEAR = eyeAspectRatio(rightEye)

                ear = (leftEAR + rightEAR) / 2

             
                imageSaved = False

                if ear < earThresh:
                    frameFlag += 1
                    print(frameFlag)
                    if frameFlag >= frameCheck:
                        print("Drowsy!!!")
                        if not imageSaved:
                            savePath = saveImage(frame)
                            image_filenames.append((savePath, formatTime(duration))) # append filename to list
                            imageSaved = True
                       
                else:
                    frameFlag = 0

                mouth = shape[mStart:mEnd]
                mar = mouthAspectRatio(mouth)
                if mar > marThresh:
                    if not imageSaved:
                            savePath = saveImage(frame)
                            image_filenames.append((savePath, formatTime(duration))) # append filename to list
                            imageSaved = True
                
        duration = int(video.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    video.release()
    cv2.destroyAllWindows()
    sorted_images = sorted(set(image_filenames), key= lambda x: x[1])
    return render_template('drowsiness.html', image_filenames=removeDuplicates(sorted_images))

app.run(debug=True)