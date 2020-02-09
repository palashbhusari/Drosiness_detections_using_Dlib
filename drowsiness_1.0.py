import cv2
import numpy as np
import dlib
import math

def distance(x,y):
    x1,y1=x  
    x2,y2=y
    cv2.circle(frame, (x1, y1), 1, (255, 0, 0), -1) # upper coordinates 
    cv2.circle(frame, (x2, y2), 1, (255, 0, 0), -1) # lower coordinates
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # dist bet 2 points
    return int(dist)
def l_eye():
    global landmarks
    global x
    global y
    # lup = left eye's upper point  and ldp left eye's lower point
    lup,ldp=(landmarks.part(37).x,landmarks.part(37).y),(landmarks.part(41).x,landmarks.part(41).y)
    #print(lup,"  ",llp)        cv2.line(frame,lup,ldp,(0,0,255),1)
    # calculating distance bet lup and ldp verticle
    l_vert = distance(lup,ldp)
    return l_vert
    

def eye_a_ratio(leye,reye):
    pass
    
    

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

        for n in range(37, 48):
            if n == 42: # skip the midle line
                continue
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            
##################################### drawing line over eyes###########################
            cv2.line(frame,(landmarks.part(n-1).x,landmarks.part(n-1).y),(x,y),(0,255,0),1)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.line(frame,(landmarks.part(36).x,landmarks.part(36).y),
                 (landmarks.part(41).x,landmarks.part(41).y),(0,255,0),1)
        cv2.line(frame,(landmarks.part(42).x,landmarks.part(42).y),
                 (landmarks.part(47).x,landmarks.part(47).y),(0,255,0),1)
######################################################################################
        
    leye=l_eye()
    #print(l_ear)
    cv2.putText(frame,str(leye),(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    if leye < 8:
        cv2.putText(frame,"drowsiness alert",(100,250), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),3,cv2.LINE_AA)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
