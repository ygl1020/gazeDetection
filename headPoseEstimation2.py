import cv2
import mediapipe as mp
import numpy as np
import time


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Counter variables for each direction
look_left_count = 0
look_right_count = 0
look_up_count = 0
look_down_count = 0

# Stage variables for tracking state
left_stage = None
right_stage = None
up_stage = None
down_stage = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1] ])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Counter logic for each direction
            # Left counter logic
            if y < -15:
                left_stage = "looking_left"
            if y > -5 and left_stage == "looking_left":
                left_stage = "center"
                look_left_count += 1

            # Right counter logic
            if y > 15:
                right_stage = "looking_right"
            if y < 5 and right_stage == "looking_right":
                right_stage = "center"
                look_right_count += 1

            # Down counter logic
            if x < -15:
                down_stage = "looking_down"
            if x > -5 and down_stage == "looking_down":
                down_stage = "center"
                look_down_count += 1

            # Up counter logic
            if x > 15:
                up_stage = "looking_up"
            if x < 5 and up_stage == "looking_up":
                up_stage = "center"
                look_up_count += 1

            # See where the user's head tilting (for display purposes)
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"
            
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Display angles next to the middle stick
            angle_text_x = int(nose_2d[0] + y * 10 + 20)
            angle_text_y = int(nose_2d[1] - x * 10)
            
            cv2.putText(image, f"x:{int(x)}", (angle_text_x, angle_text_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"y:{int(y)}", (angle_text_x, angle_text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"z:{int(z)}", (angle_text_x, angle_text_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add the direction text at the top
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Draw counter boxes in top right corner (smaller size)
            box_width = 80
            box_height = 50
            start_x = img_w - 180  # Starting from right side
            start_y = 10
            
            # Left counter box
            cv2.rectangle(image, (start_x, start_y), (start_x + box_width, start_y + box_height), (245, 117, 16), -1)
            cv2.putText(image, 'LEFT', (start_x + 15, start_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, str(look_left_count), 
                        (start_x + 25, start_y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Right counter box
            cv2.rectangle(image, (start_x + 90, start_y), (start_x + 90 + box_width, start_y + box_height), (245, 117, 16), -1)
            cv2.putText(image, 'RIGHT', (start_x + 100, start_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, str(look_right_count), 
                        (start_x + 115, start_y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Up counter box
            cv2.rectangle(image, (start_x, start_y + 60), (start_x + box_width, start_y + 60 + box_height), (245, 117, 16), -1)
            cv2.putText(image, 'UP', (start_x + 25, start_y + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, str(look_up_count), 
                        (start_x + 25, start_y + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Down counter box
            cv2.rectangle(image, (start_x + 90, start_y + 60), (start_x + 90 + box_width, start_y + 60 + box_height), (245, 117, 16), -1)
            cv2.putText(image, 'DOWN', (start_x + 102, start_y + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, str(look_down_count), 
                        (start_x + 115, start_y + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        end = time.time()
        totalTime = end - start
        
        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20, img_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        try:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    
            cv2.imshow('Head Pose Estimation', image)
        except Exception as e:
            print(f"Error: {e}")
    else:
        cv2.imshow('Head Pose Estimation', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()