import numpy as np
from scipy import ndimage as ndimage
import cv2
import mediapipe as mp
import torch
from time import time
import matplotlib.pyplot as plt
from model_skeletal import HandGestureNet
import itertools


# Use Mediapipe to find hand landmarks
# Reorder landmarks
# Find 22nd landmark
# Make correct datasize (100)
# Throw into model
# See what happens

num_classes = 14
num_channels = 66
sequence_length = 20
input_data = []
value = 0
out = np.zeros(num_classes)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


## --------------------------------------------------------------

## INITALIZE MODEL ----------------------------
model = HandGestureNet(n_channels=num_channels, n_classes=num_classes)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('gesture_pretrained_model_14c_80_3-14.pt'))
else:
    model.load_state_dict(torch.load('gesture_pretrained_model_14c_80_3-14.pt', map_location=torch.device('cpu')))

model = model.float()
model.eval()
## ---------------------------------------------


def find_coords(img, results):

    lmlist = []
   
    myHand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myHand.landmark):
        # print(lm)
        lmlist.append([lm.x, lm.y, lm.z])

    return lmlist

def find_hand(frame):
        with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.35, min_tracking_confidence=0.35, max_num_hands=1) as hands:
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
               
                # print(results.multi_hand_landmarks)
                h, w, c = frame.shape

                lmlist = find_coords(frame, results)
                # print(lmlist)

                hand_center = []
                center_x = (lmlist[0][0] + lmlist[9][0])/2
                center_y = (lmlist[0][1] + lmlist[9][1])/2
                center_z = (lmlist[0][2] + lmlist[9][2])/2
                hand_center = [center_x, center_y, center_z]
                # print(hand_center)
                # print(lmlist)
                lmlist.insert(1,hand_center)
                frame = cv2.circle(frame, (int(hand_center[0]*w), int(hand_center[1]*h)), 3, (255,0,0), 2)

                lmlist = list(itertools.chain.from_iterable(lmlist))

                return lmlist
            
def resize_gestures(input_gestures, final_length=100):
    """
    Resize the time series by interpolating them to the same length

    Input:
        - input_gestures: list of numpy.ndarray tensors.
              Each tensor represents a single gesture.
              Gestures can have variable durations.
              Each tensor has a shape: (duration, channels)
              where duration is the duration of the individual gesture
                    channels = 44 = 2 * 22 if recorded in 2D and
                    channels = 66 = 3 * 22 if recorded in 3D 
    Output:
        - output_gestures: one numpy.ndarray tensor.
              The output tensor has a shape: (records, final_length, channels)
              where records = len(input_gestures)
                   final_length is the common duration of all gestures
                   channels is the same as above 
    """

    print(input_gestures.shape)
    # please use python3. if you still use python2, important note: redefine the classic division operator / by importing it from the __future__ module
    output_gestures = np.array([np.array([ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(np.size(x_i, 1))]).T for x_i in input_gestures])
    return output_gestures

camera = cv2.VideoCapture("/dev/video0")
camera.set(cv2.CAP_PROP_FPS, 48)


while(True):

    ret, frame = camera.read()

    hand = find_hand(frame)


    if len(input_data) < sequence_length and hand is not None:
        input_data.append(hand)
        # print(len(input_data))


    # color_image = cv2.rectangle(frame, (0,0), (640,480), (255, 0,0), 2)

    cv2.imshow("frame", frame)

    if len(input_data) == sequence_length:
    # Make infrence

        with torch.no_grad():
            # print(test.shape)
            print("TIME TO MAKE AN INFRENCE")
            input_data = np.array(input_data)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = resize_gestures(input_data)

            input_data = torch.from_numpy(input_data).float()

            print(input_data.shape)

            prediction = model(input_data)
            print(prediction)
            _, prediction = prediction.max(dim=1)
            print("predicted {}".format(prediction.tolist()))

            input_data = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("done?")
camera.release()
cv2.destroyAllWindows()