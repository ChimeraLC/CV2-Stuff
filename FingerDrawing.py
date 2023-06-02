# Temporary incomplete project

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
	total = 0
	for i in range(len(vector1)):
		total += vector1[i] * vector2[i]
	return total

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
		raised = True
		for handslms in result.multi_hand_landmarks:
			for lm in handslms.landmark:
				# print(id, lm)
				lmx = int(lm.x * 640)
				lmy = int(lm.y * 480)
				landmarks.append([lmx, lmy])
			
				
		
		# highlight stuff
		if (raised):
			for i in range(5, 9):
				cv2.circle(frame, landmarks[i], 2,
					    (50,152,152), 2)

		# Drawing landmarks on frames
		mpDraw.draw_landmarks(frame, handslms, 
		    mpHands.HAND_CONNECTIONS)
		
		for i in range(5, 6):
			term1 = handslms.landmark[i]
			term2 = handslms.landmark[i+1]
			term3 = handslms.landmark[i+2]
			cv2.circle(frame, (int(term1.x * 640), int(term1.y * 480)), 2,
					(50,152,152), 2)
			cv2.putText(frame,
	          "[" + str(int(100 * (term1.x-term2.x))) + " , " + str(int(100*(term1.y - term2.y))) + 
			  " , " + str(int(100*(term1.z - term2.z))) + "]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

		

		
		

	# Show the final output
	cv2.imshow("Output", frame) 

	if cv2.waitKey(1) == ord(' '):
		break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()