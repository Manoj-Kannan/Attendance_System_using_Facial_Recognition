import os
import numpy as np
import cv2
from datetime import datetime
import face_recognition

def imgEncode(images):
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeImg.append(encode)
    return encodeImg

def markAttendance(name):
    nameList = []
    with open('../attendance.csv','r+') as f:
        entries = f.readlines()
        for line in entries:
            entry = line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

images = []
classNames = []
encodeImg = []

#Load Source Images
for img in os.listdir(os.chdir('img_source')):
    curImg = cv2.imread(img)
    images.append(curImg)
    classNames.append(img[:-4])

# return the 128-dimension face encoding for each face in the image
srcImgEncode = imgEncode(images)
print('Encoding Complete')

# define a video capture object
vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print("Cannot open camera")
    exit()
while True:
    success, live = vid.read()
    # resize the images for faster processing
    liveResized = cv2.resize(live,(0,0),None, 0.25, 0.25)
    # cv2.imread() - images with channels stored in B G R order
    # Convert it into R G B which is the preferred channel for face_recognition
    liveResized = cv2.cvtColor(liveResized,cv2.COLOR_BGR2RGB)

    # return (top, right, bottom, left) - pixel values of face locations  
    facesLocLive = face_recognition.face_locations(liveResized)
    # return the 128-dimension face encoding for each face in the image
    facesEncodeLive = face_recognition.face_encodings(liveResized)

    for faceLoc, faceEncode in zip(facesLocLive, facesEncodeLive):
        # Compare a list of face encodings against a candidate encoding to see if they match
        matches = face_recognition.compare_faces(srcImgEncode, faceEncode)
        # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. 
        # The distance tells you how similar the faces are.
        distances = face_recognition.face_distance(srcImgEncode, faceEncode)
        # find the sample with least distance (signifies similarity in faces)
        matchIndex = np.argmin(distances)
        if matches[matchIndex] == True:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            # previously, resized the images for faster processing.
            # now, for visualization resize back to actual form.
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # create bounding box around the face
            cv2.rectangle(live, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=1)
            cv2.rectangle(live, (x1,y2),(x2,y2+35),(0,255,0), cv2.FILLED)
            cv2.putText(live, name.split()[0], (x1+6,y2+25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)
            
    
    cv2.imshow('Live', live)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
