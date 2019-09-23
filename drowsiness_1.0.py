import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        hull_points=[]      # list to append coordinates and fill color
        for n in range(37, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            hull_points.append([x,y]) 
                # now drawing line for lips
            cv2.line(frame,(landmarks.part(n-1).x,landmarks.part(n-1).y),(x,y),(0,255,0),1)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                
        pts=np.asarray([hull_points])
        #cv2.fillPoly(frame, pts, color=(0,0,255)) # fil color in the np list hull_pts
    #print("hull = ",pts, type(pts))
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
