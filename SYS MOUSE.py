import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Function to draw points on finger tips
def draw_finger_points(frame, hand_landmarks):
    h, w, _ = frame.shape
    for landmark in hand_landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)  # Green points

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to control the mouse based on hand gestures
def control_mouse(hand_landmarks, prev_index_pos):
    if hand_landmarks:
        # Get landmarks for fingers
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()

        # Calculate current positions
        screen_thumb_x, screen_thumb_y = thumb_tip.x * screen_width, thumb_tip.y * screen_height
        screen_index_x, screen_index_y = index_tip.x * screen_width, index_tip.y * screen_height
        screen_pinky_x, screen_pinky_y = pinky_tip.x * screen_width, pinky_tip.y * screen_height

        if prev_index_pos is not None:
            # Calculate movement delta
            delta_x = screen_index_x - prev_index_pos[0]
            delta_y = screen_index_y - prev_index_pos[1]

            # Move the system cursor
            current_mouse_x, current_mouse_y = pyautogui.position()
            new_mouse_x = max(0, min(screen_width - 1, current_mouse_x + delta_x))
            new_mouse_y = max(0, min(screen_height - 1, current_mouse_y + delta_y))

            pyautogui.moveTo(new_mouse_x, new_mouse_y)

        # Check for clicks based on proximity
        thumb_index_dist = distance((screen_thumb_x, screen_thumb_y), (screen_index_x, screen_index_y))
        thumb_pinky_dist = distance((screen_thumb_x, screen_thumb_y), (screen_pinky_x, screen_pinky_y))

        if thumb_index_dist < 50:  # Distance threshold for click
            pyautogui.click()
            print("Left click detected!")

        if thumb_pinky_dist < 50:  # Distance threshold for right-click
            pyautogui.rightClick()
            print("Right click detected!")

        # Update previous index position
        return (screen_index_x, screen_index_y)

    return prev_index_pos

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

prev_index_pos = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally to simulate a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw hand landmarks and finger points
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            draw_finger_points(frame, hand_landmarks)
            prev_index_pos = control_mouse(hand_landmarks, prev_index_pos)

    # Display the frame
    cv2.imshow('Hand Gesture Control', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
