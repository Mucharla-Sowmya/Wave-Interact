import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import openai
import time

# MediaPipe Hand Module Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Streamlit app
st.title("Gesture-Based Human-Computer Interaction System")
st.markdown("Use hand gestures to interact with the system. Try gestures like 'thumbs up', 'fist', 'okay', or 'peace'.")

# Streamlit video capture
camera = st.camera_input("Capture Hand Gestures")

# OpenAI API Setup (make sure you have an OpenAI API key)
openai.api_key = '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'

# Placeholder for AI-generated output
output_placeholder = st.empty()

# Function to map gestures to simple descriptions
def get_gesture_description(hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        
        # Extract finger tips and joints
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
            return "Fist"
        
        # Check for "Thumbs Up" gesture: Thumb tip above the index tip and other fingers curled
        elif (thumb_tip.y < index_tip.y and 
              index_tip.y > index_dip.y and 
              middle_tip.y > middle_dip.y and 
              ring_tip.y > ring_dip.y and 
              pinky_tip.y > pinky_dip.y):
            return "Thumbs Up"
        
        # Check for "Open Hand" gesture: All fingers extended
        elif (index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and
              thumb_tip.y < thumb_ip.y and
              thumb_tip.x > index_tip.x and 
              index_tip.y < index_dip.y and
              middle_tip.y < middle_dip.y and
              ring_tip.y < ring_dip.y and
              pinky_tip.y < pinky_dip.y):
            return "Open Hand"
        
        # Check for "Okay" gesture: Thumb and index tip are close, others extended
        elif (abs(thumb_tip.x - index_tip.x) < 0.05 and
              abs(thumb_tip.y - index_tip.y) < 0.05 and
              middle_tip.y < middle_dip.y and
              ring_tip.y < ring_dip.y and
              pinky_tip.y < pinky_dip.y):
            return "Okay"
        
        # Check for "Rock" gesture (Sign of the Horns): Index and pinky fingers extended, others curled
        elif (index_tip.y < index_dip.y and 
              pinky_tip.y < pinky_dip.y and
              abs(index_tip.x - pinky_tip.x) > 0.2 and 
              middle_tip.y > middle_dip.y and
              ring_tip.y > ring_dip.y):
            return "Rock"
        
        # Check for "Peace" gesture: Index and middle fingers extended, others curled
        elif (index_tip.y < index_dip.y and 
              middle_tip.y < middle_dip.y and
              abs(index_tip.x - middle_tip.x) < 0.1 and 
              ring_tip.y > ring_dip.y and 
              pinky_tip.y > pinky_dip.y):
            return "Peace"
        
        # Check for "Love" gesture: Index, pinky, and thumb extended, others curled
        elif (index_tip.y < index_dip.y and
              pinky_tip.y < pinky_dip.y and
              thumb_tip.y < thumb_ip.y and
              abs(index_tip.x - pinky_tip.x) > 0.2 and 
              middle_tip.y > middle_dip.y and
              ring_tip.y > ring_dip.y):
            return "Love"
        
    return "No gesture detected."

# Function to get a natural language response from OpenAI based on gesture description
def generate_openai_response(gesture_description):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Change this to whichever model you are using
            messages=[{
                "role": "system", "content": "You are a helpful assistant."
            }, {
                "role": "user", "content": gesture_description
            }]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

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

        # Get gesture description based on detected hand landmarks
        gesture_description = get_gesture_description(results.multi_hand_landmarks)

        # Generate a response using OpenAI
        ai_response = generate_openai_response(gesture_description)

        # Display the result on the Streamlit app
        output_placeholder.text(f"Detected Gesture: {gesture_description}\n")
    else:
        output_placeholder.text("No hands detected. Please show your hand in front of the camera.")

    # Convert frame to Streamlit-friendly format (Display in Streamlit)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame)

# Clean up
hands.close()
