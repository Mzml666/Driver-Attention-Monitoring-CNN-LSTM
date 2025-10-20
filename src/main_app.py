import cv2
import mediapipe as mp
import math
import numpy as np # --- NEW ---

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Helper function to calculate Eye Aspect Ratio (EAR)
def get_ear(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Landmark indices for left and right eyes
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EVE_INDICES = [33, 160, 158, 133, 153, 144] # Corrected variable name

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# --- NEW: Head Pose Estimation Setup ---
# A standard 3D model of a face
face_3d_model_points = np.array([
    [0.0, 0.0, 0.0],     # Nose tip
    [0.0, -330.0, -65.0], # Chin
    [-225.0, 170.0, -135.0], # Left eye left corner
    [225.0, 170.0, -135.0],  # Right eye right corner
    [-150.0, -150.0, -125.0], # Left Mouth corner
    [150.0, -150.0, -125.0]  # Right mouth corner
])
# Indices for the 6 landmarks above
face_2d_model_indices = [1, 152, 263, 33, 291, 61] 

# --- END NEW ---

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape

        # --- NEW: Camera Matrix for solvePnP ---
        # A simplified camera matrix (assuming no lens distortion)
        focal_length = image_width
        cam_center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, cam_center[0]],
            [0, focal_length, cam_center[1]],
            [0, 0, 1]
        ], dtype=np.double)
        dist_coeffs = np.zeros((4, 1)) # Assuming no distortion
        # --- END NEW ---

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        drowsy_alert = False # --- NEW: Flag for alerts

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

                landmarks_list = []
                for landmark in face_landmarks.landmark:
                    landmarks_list.append((int(landmark.x * image_width), int(landmark.y * image_height)))

                # --- 1. Drowsiness Detection ---
                left_ear = get_ear(landmarks_list, LEFT_EYE_INDICES)
                right_ear = get_ear(landmarks_list, RIGHT_EVE_INDICES) # Corrected variable name
                avg_ear = (left_ear + right_ear) / 2.0

                cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                EAR_THRESHOLD = 0.20 
                if avg_ear < EAR_THRESHOLD:
                    drowsy_alert = True # Set the flag

                # --- 2. Distraction Detection (Head Pose) ---
                face_2d_points = np.array([
                    landmarks_list[idx] for idx in face_2d_model_indices
                ], dtype=np.double)

                # Solve for pose
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    face_3d_model_points, face_2d_points, camera_matrix, dist_coeffs
                )
                
                # Convert rotation vector to angles (Pitch, Yaw, Roll)
                (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                # Display angles
                cv2.putText(image, f"Pitch: {pitch:.1f}", (350, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f"Yaw: {yaw:.1f}", (350, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # --- NEW: Distraction Alert Logic ---
                # You will need to TUNE these thresholds
                YAW_THRESHOLD = 25  
                PITCH_THRESHOLD = 20

                if yaw > YAW_THRESHOLD or yaw < -YAW_THRESHOLD or pitch > PITCH_THRESHOLD or pitch < -PITCH_THRESHOLD:
                    cv2.putText(image, "DISTRACTION DETECTED!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # --- END NEW ---

                # --- NEW: Display Drowsiness Alert ---
                if drowsy_alert:
                    cv2.putText(image, "DROWSINESS DETECTED!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # --- END NEW ---


        cv2.imshow('Driver Attention Monitoring', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()