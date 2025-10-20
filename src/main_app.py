import cv2
import mediapipe as mp
import math # --- NEW: Import math for calculations

# --- NEW: Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- NEW: Helper function to calculate Eye Aspect Ratio (EAR)
def get_ear(landmarks, eye_indices):
    # Get the (x, y) coordinates for the six eye landmarks
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    # Calculate the vertical distances
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    # Calculate the horizontal distance
    horizontal = euclidean_distance(p1, p4)

    # Calculate the EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# --- NEW: Landmark indices for left and right eyes
# These indices are specific to Mediapipe's 478-landmark model
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Initialize Mediapipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start the webcam
cap = cv2.VideoCapture(0)

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
        
        # --- FLIP THE IMAGE HERE ---
        image = cv2.flip(image, 1)

        # Get image dimensions
        image_height, image_width, _ = image.shape

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Draw the face mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

                # --- NEW: Calculate EAR ---
                # We need to convert normalized (0.0-1.0) landmarks to pixel coordinates
                landmarks_list = []
                for landmark in face_landmarks.landmark:
                    landmarks_list.append((int(landmark.x * image_width), int(landmark.y * image_height)))

                # Calculate EAR for both eyes
                left_ear = get_ear(landmarks_list, LEFT_EYE_INDICES)
                right_ear = get_ear(landmarks_list, RIGHT_EYE_INDICES)

                # Average the EAR for a more stable result
                avg_ear = (left_ear + right_ear) / 2.0

                # Display the EAR on the screen
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # --- NEW: Simple Drowsiness Detection Logic ---
                # We'll define a threshold. If EAR drops below it, the person is drowsy.
                # You will need to TUNE this threshold.
                EAR_THRESHOLD = 0.20 
                
                if avg_ear < EAR_THRESHOLD:
                    cv2.putText(image, "DROWSINESS DETECTED!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow('Driver Attention Monitoring', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()