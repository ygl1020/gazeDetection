import cv2
import mediapipe as mp
import numpy as np
my_drawing = mp.solutions.drawing_utils
my_pose = mp.solutions.pose

pose = my_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) # set up mediapipe instance to access pose model, tradeoff
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     try:
#         # read the real time frame, frame is real time feed 
#         ret,frame = cap.read()
#         # recolor image since the mp image use rgb format
#         image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
        
#         # make prediction
#         results = pose.process(image)
#         # print(results)
#         # recolor back to bgr
#         image.flags.writeable=True
#         image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
   
#         # render detections
#         my_drawing.draw_landmarks(image,results.pose_landmarks,my_pose.POSE_CONNECTIONS,
#                                   my_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2 ),
#                                   my_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2 ))
#         image = cv2.flip(image, 1)
        
#         # print(1)
#         cv2.imshow("live image",image)
#         if cv2.waitKey(10) & 0xFF== 27:
#             print("Escape key pressed. Exiting.")
#             break
#         # print(2)
#     except Exception as e:
#         print(f"An error occurred: {e}")
# cap.release()
# cv2.destroyAllWindows()

#--------------------------------------
# extract the landmart
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     try:
#         # read the real time frame, frame is real time feed 
#         ret,frame = cap.read()
#         # recolor image since the mp image use rgb format
#         image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
        
#         # make prediction
#         results = pose.process(image)
#         # print(results)
#         # recolor back to bgr
#         image.flags.writeable=True
#         image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

#         # extract landmarks, use try to aviod the corruption when web cam has issue
#         my_pose.PoseLandmark
#         try:
#             landmarks = results.pose_landmarks.landmark
#             print(landmarks)
#         except:
#             pass
#         # render detections
#         my_drawing.draw_landmarks(image,results.pose_landmarks,my_pose.POSE_CONNECTIONS,
#                                   my_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2 ),
#                                   my_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2 ))
#         image = cv2.flip(image, 1)
        
#         # print(1)
#         cv2.imshow("live image",image)
#         if cv2.waitKey(10) & 0xFF== 27:
#             print("Escape key pressed. Exiting.")
#             break
#         # print(2)
#     except Exception as e:
#         print(f"An error occurred: {e}")
# cap.release()
# cv2.destroyAllWindows()
for lndmark in my_pose.PoseLandmark:
    print(lndmark, lndmark.value)
    
    
    
