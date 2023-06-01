# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mediapipe.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

#helper functions
def DotProduct(vector1, vector2):
	return vector1[0] * vector2[0] + vector1[1] * vector2[1]

while True:
	# Read each frame from the webcam
	_, frame = cap.read()

	x, y, c = frame.shape

	# Flip the frame vertically
	frame = cv2.flip(frame, 1)
	framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Get hand landmark prediction
	result = hands.process(framergb)

	className = ''

	# post process the result
	if result.multi_hand_landmarks:
		landmarks = []
		for handslms in result.multi_hand_landmarks:
			for lm in handslms.landmark:
				# print(id, lm)
				lmx = int(lm.x * 640)
				lmy = int(lm.y * 480)
				landmarks.append([lmx, lmy])
				
				#cv2.circle(frame, (lmx, lmy), 1, (50, 152, 152), 2);

		# Drawing landmarks on frames
		mpDraw.draw_landmarks(frame, handslms, 
		    mpHands.HAND_CONNECTIONS)

		# Detecting direction hand is pointing
		base = landmarks[0]
		top = [int(landmarks[5][0]/2 + landmarks[17][0]/2),
	 	    int(landmarks[5][1]/2 + landmarks[17][1]/2)]
		cv2.line(frame, base, top,  (0,0,0), 3)

		# Checking finger directions
		raised = 0
		baseVector = [top[0] - base[0], top[1] - base[1]]
		for i in range(8, 21, 4):
			extended = True
			for j in range(0, 3):
				currentVector = [landmarks[i-j][0] - landmarks[i-j-1][0], 
		     		    landmarks[i-j][1] - landmarks[i-j-1][1]]
				if (DotProduct(currentVector, baseVector) < 0):
					extended = False
			if (extended):
				for j in range(0, 4):
					cv2.circle(frame, landmarks[i-j], 2,
					    (50,152,152), 2)
				raised += 1
		
		cv2.putText(frame, str(raised), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

	# Show the final output
	cv2.imshow("Output", frame) 

	if cv2.waitKey(1) == ord(' '):
		break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()