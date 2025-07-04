import cv2
import numpy as np
import dlib
from math import hypot
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def mid_point(p1,p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

def get_blinking_ratio(eye_points,facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = mid_point(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_botton = mid_point(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    
    # draw the lines at eyes
    # hor_line = cv2.line(frame,left_point,right_point,(0,255,0),2)
    # ver_line = cv2.line(frame,center_top,center_botton,(0,255,0),2)
    
    # calculate ratio 
    hor_line_length = hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    ver_line_lenght = hypot((center_top[0]-center_botton[0]),(center_top[1]-center_botton[1]))
    ratio = hor_line_length/ver_line_lenght
    return ratio
while cap:
    res,frame = cap.read()
    # dlib use gray rag 
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # make face object detection
    faces = detector(gray)
    
    # for each face, extract out the four pointer position, so we can calculate the ratio betweem two line to calculate blink  time
    for face in faces:
        # x,y = face.left(), face.top()
        # x1,y1 = face.right(),face.bottom()
        # cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        # print(face, face.left(), face.top(), face.right(),face.bottom())
        # detect blinking
        landmarks = predictor(gray,face)
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2
        if blinking_ratio <=3.5:
            cv2.putText(frame,"BLIKING", (50,150), 1,4,(255,0,0))
        
        # gaze detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)],np.int32)
        # cv2.circle(frame, (x,y),3,(0,0,255),2)
        # draw the eye outside line
        # cv2.polylines(frame,[left_eye_region],True, (0,0,225),2)
        min_x = np.min(left_eye_region[:,0])
        max_x = np.max(left_eye_region[:,0])
        min_y = np.min(left_eye_region[:,1])
        max_y = np.max(left_eye_region[:,1])
        eye = frame[min_y:max_y, min_x:max_x]
        eye = cv2.resize(eye,None,-1,fx=5,fy=5)
        cv2.imshow("Eye",eye)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
# print(faces)
cap.release()
cv2.destroyAllWindows()