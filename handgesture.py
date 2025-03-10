import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import google.generativeai as genai

# MediaPipe Hand Module Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Streamlit app
st.title("Gesture-Based Human-Computer Interaction System")
st.markdown("Use hand gestures to interact with the system. Try gestures like 'thumbs up', 'fist', 'okay', 'peace', or 'heart'.")

# Streamlit video capture
camera = st.camera_input("Capture Hand Gestures")

# Configure Gemini API
genai.configure(api_key='AIzaSyDC-DZY-_trmckvcwm_N8uga8zzDX0YWVI')  # Replace with your Gemini API key

# Placeholder for AI-generated output
output_placeholder = st.empty()

# Function to map gestures to simple descriptions and determine left or right hand
def get_gesture_description_and_hand_side(hand_landmarks, frame_width):
    if hand_landmarks:
        # Assuming we are detecting one hand
        landmarks = hand_landmarks[0].landmark

        # Calculate the average x-coordinate of the hand landmarks
        avg_x = np.mean([landmark.x for landmark in landmarks])
        
        # Determine if it is left or right hand
        hand_side = "Right" if avg_x < 0.5 else "Left"  # Corrected here

        # Check for heart gesture (two hands)
        if len(hand_landmarks) == 2:  # Check if both hands are detected
            left_hand = hand_landmarks[0].landmark
            right_hand = hand_landmarks[1].landmark

            # Extract key landmarks
            left_index_tip = left_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_thumb_tip = left_hand[mp_hands.HandLandmark.THUMB_TIP]
            
            right_index_tip = right_hand[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_thumb_tip = right_hand[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate distances between fingertips
            index_distance = ((left_index_tip.x - right_index_tip.x) ** 2 + 
                             (left_index_tip.y - right_index_tip.y) ** 2) ** 0.5
            thumb_distance = ((left_thumb_tip.x - right_thumb_tip.x) ** 2 + 
                             (left_thumb_tip.y - right_thumb_tip.y) ** 2) ** 0.5

            # Check if fingertips are close enough to form a heart
            if index_distance < 0.1 and thumb_distance < 0.1:  # Adjust thresholds as needed
                return "Heart", "Both hands"

        # Single-hand gestures
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
        
        # Check for "Fist" gesture: All fingers curled
        if (thumb_tip.y > thumb_ip.y and
            index_tip.y > index_dip.y and
            middle_tip.y > middle_dip.y and
            ring_tip.y > ring_dip.y and
            pinky_tip.y > pinky_dip.y):
            return "Fist", hand_side
        
        # Check for "Thumbs Up" gesture: Thumb tip above the index tip and other fingers curled
        elif (thumb_tip.y < index_tip.y and 
              index_tip.y > index_dip.y and 
              middle_tip.y > middle_dip.y and 
              ring_tip.y > ring_dip.y and 
              pinky_tip.y > pinky_dip.y):
            return "Thumbs Up", hand_side
        
        # Check for "Open Hand" gesture: All fingers extended
        elif (thumb_tip.y < thumb_ip.y and  # Thumb extended
              index_tip.y < index_dip.y and  # Index finger extended
              middle_tip.y < middle_dip.y and  # Middle finger extended
              ring_tip.y < ring_dip.y and  # Ring finger extended
              pinky_tip.y < pinky_dip.y):  # Pinky finger extended
            return "Open Hand", hand_side
        
        # Check for "Okay" gesture: Thumb and index tip are close, others extended
        elif (abs(thumb_tip.x - index_tip.x) < 0.05 and
              abs(thumb_tip.y - index_tip.y) < 0.05 and
              middle_tip.y < middle_dip.y and
              ring_tip.y < ring_dip.y and
              pinky_tip.y < pinky_dip.y):
            return "Okay", hand_side
        
        # Check for "Rock" gesture (Sign of the Horns): Index and pinky fingers extended, others curled
        elif (index_tip.y < index_dip.y and 
              pinky_tip.y < pinky_dip.y and
              middle_tip.y > middle_dip.y and
              ring_tip.y > ring_dip.y):
            return "Rock", hand_side
        
        # Check for "Peace" gesture: Index and middle fingers extended, others curled
        elif (index_tip.y < index_dip.y and 
              middle_tip.y < middle_dip.y and
              abs(index_tip.x - middle_tip.x) < 0.1 and 
              ring_tip.y > ring_dip.y and 
              pinky_tip.y > pinky_dip.y):
            return "Peace", hand_side
        
        # Check for "Love" gesture: Thumb, index, and pinky extended; middle and ring fingers curled
        elif (thumb_tip.y < thumb_ip.y and  # Thumb extended
              index_tip.y < index_dip.y and  # Index finger extended
              pinky_tip.y < pinky_dip.y and  # Pinky finger extended
              middle_tip.y > middle_dip.y and  # Middle finger curled
              ring_tip.y > ring_dip.y):  # Ring finger curled
            return "Love", "Both hands"
        
    return "No gesture detected.", "Unknown"

# Function to get a natural language response from Gemini based on gesture description
def generate_gemini_response(gesture_description, hand_side):
    try:
        # Provide gesture-specific responses
        if gesture_description == "Thumbs Up":
            return f"Awesome! You're on the right track with your {hand_side} hand! Keep up the good work! 👍"
        elif gesture_description == "Fist":
            return f"You're showing strength with your {hand_side} hand! Keep it up! 💪"
        elif gesture_description == "Open Hand":
            return f"Your {hand_side} hand is showing calm and peace. Relax, you've got this! ✋"
        elif gesture_description == "Okay":
            return f"Everything's good with your {hand_side} hand! Let's move ahead with confidence. 👍"
        elif gesture_description == "Rock":
            return f"Rock on with your {hand_side} hand! Stay rebellious and have fun! 🤘"
        elif gesture_description == "Peace":
            return f"Your {hand_side} hand is sending peace and love to the world! ✌️"
        elif gesture_description == "Heart":
            return f"Spreading love with your both hands! Keep those positive vibes flowing! ❤️"
        else:
            return f"Sorry, no gesture detected with your {hand_side} hand. Try again! ✋"
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, something went wrong with the AI response."

# Instantiate the hands object
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Main loop for real-time hand gesture detection
if camera:
    # Read the image from the uploaded file
    frame_bytes = camera.read()  # Read the byte data from the camera input

    # Convert the byte data to a NumPy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    
    # Decode the image from the byte array
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame to detect hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get gesture description and hand side based on detected hand landmarks
        gesture_description, hand_side = get_gesture_description_and_hand_side(results.multi_hand_landmarks, frame.shape[1])

        # Generate a response using Gemini
        ai_response = generate_gemini_response(gesture_description, hand_side)

        # Display the result on the Streamlit app
        output_placeholder.text(f"Detected Gesture: {gesture_description} ({hand_side} )\nAI Response: {ai_response}")
    else:
        output_placeholder.text("No hands detected. Please show your hand in front of the camera.")

    # Convert frame to Streamlit-friendly format (Display in Streamlit)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame)

# Clean up
hands.close()
