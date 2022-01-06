import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np


def Images_Recognition():
    lst = os.listdir('Images')
    imgs = []
    faceLoc = []
    faceEncode = []
    results = []
    diff = []

    for img in lst:
        i = face_recognition.load_image_file(f'Images/{img}')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        imgs.append(i)
        l = face_recognition.face_locations(i)[0]
        e = face_recognition.face_encodings(i)[0]
        faceLoc.append(l)
        faceEncode.append(e)

    for i in range(1, len(lst)):
        r = face_recognition.compare_faces([faceEncode[0]], faceEncode[i])
        results.append(r)
        d = face_recognition.face_distance([faceEncode[0]], faceEncode[i])
        diff.append(d)

    for i in range(len(lst)):
        cv2.rectangle(imgs[i], (faceLoc[i][3], faceLoc[i][0]), (faceLoc[i][1], faceLoc[i][2]), (203, 192, 255), 2)

    for i in range(1, len(lst)):
        cv2.putText(imgs[i], f'{results[i - 1][0]} {round((1 - round(diff[i - 1][0], 4)) * 100, 4)} %',
                    (faceLoc[i][3] - 20, faceLoc[i][0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for i in range(len(lst)):
        cv2.imshow(f'{i}', imgs[i])

    cv2.waitKey()


def Cam_Recognition():
    lst = os.listdir('ImgsTrain')
    knownEncode = []
    imgs = []
    classNames = []

    # save results
    def saveResults(name):
        with open('results.csv', 'r+') as f:
            results = f.readlines()
            nameList = []
            for line in results:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    for img in lst:
        i = face_recognition.load_image_file(f'ImgsTrain/{img}')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        imgs.append(i)
        e = face_recognition.face_encodings(i)[0]
        knownEncode.append(e)
        classNames.append(os.path.splitext(img)[0])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
        frame0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faceCurrentLoc = face_recognition.face_locations(frame0)
        faceCurrentEncode = face_recognition.face_encodings(frame0)

        for faceEncode, faceLoc in zip(faceCurrentEncode, faceCurrentLoc):
            results = face_recognition.compare_faces(knownEncode, faceEncode)
            diff = face_recognition.face_distance(knownEncode, faceEncode)
            matchIndex = np.argmin(diff)

            if diff[matchIndex] < 0.5:
                name = classNames[matchIndex].upper()
                saveResults(name)
            else:
                name = 'Unknown'

            # putText
            # draw Rectangle around faces
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (203, 192, 255, 0), 2)
            cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Images_Recognition()
Cam_Recognition()
