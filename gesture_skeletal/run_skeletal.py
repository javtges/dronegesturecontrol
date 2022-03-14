import numpy as np
from scipy import ndimage as ndimage
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import torch
from time import time
import matplotlib.pyplot as plt
from model_skeletal import HandGestureNet


# Use Mediapipe to find hand landmarks
# Reorder landmarks
# Find 22nd landmark
# Make correct datasize (100)
# Throw into model
# See what happens

num_classes = 14
num_channels = 66
sequence_length = 60
input_data = []
value = 0
out = np.zeros(num_classes)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

## REALSENSE INIT ----------------------------------
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.color)
intr = profile.as_video_stream_profile().get_intrinsics()

# profile = config.get_stream(rs.stream.depth)

align_to = rs.stream.color
align = rs.align(align_to)

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


def get_xyz_from_image(depth_image, x, y, intr):
    # print(depth_image.shape)
    # HERE, Y IS ROWS AND X IS COLUMNS
    # print(x, y)
    # print(depth_image[y,x])

    result = rs.rs2_deproject_pixel_to_point(intr, [y,x], depth_image[y,x])
    # print(result)
    ret_val = [0, 0, 0]
    ret_val[0] = result[1]/1000
    ret_val[1] = result[0]/1000
    ret_val[2] = result[2]/1000
    return ret_val

def find_coords(img, results):

    lmlist = []
   
    myHand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmlist.append([cx, cy])

    return lmlist

def find_hand(frame, depth_image):
        with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.35, min_tracking_confidence=0.35, max_num_hands=1) as hands:
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
               
                # print(results.multi_hand_landmarks)

                lmlist = find_coords(frame, results)
                # print(lmlist)

                hand_center = []
                center_x = (lmlist[0][0] + lmlist[9][0])/2
                center_y = (lmlist[0][1] + lmlist[9][1])/2
                hand_center = [int(center_x), int(center_y)]
                # print(hand_center)
                # print(lmlist)
                lmlist.insert(1,hand_center)
                frame = cv2.circle(frame, (hand_center[0], hand_center[1]), 3, (255,0,0), 2)
                # print(len(lmlist))
                # print(lmlist)
                distance_list = []
                
                for point in lmlist:
                    if point[1] >= 480:
                        point[1] = 479
                    if point[0] >= 640:
                        point[0] = 639
                    # POINT 0 IS X POINT 1 IS Y
                    coords = get_xyz_from_image(depth_image, point[0], point[1], intr)
                    distance_list = distance_list + coords

                # print(distance_list)
                # print(len(distance_list))

                return distance_list
            
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
    # please use python3. if you still use python2, important note: redefine the classic division operator / by importing it from the __future__ module
    output_gestures = np.array([np.array([ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(np.size(x_i, 1))]).T for x_i in input_gestures])
    return output_gestures

try:
    while(True):

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # print(color_image.shape)
        # print(depth_image.shape)

        try:
            hand = find_hand(color_image, depth_image)

            if len(input_data) < sequence_length and hand is not None:
                input_data.append(hand)
                # print(len(input_data))
        except:
            pass

        color_image = cv2.rectangle(color_image, (0,0), (640,480), (255, 0,0), 2)

        cv2.imshow("frame", color_image)

        if len(input_data) == sequence_length:
        # Make infrence

            with torch.no_grad():
                print(len(input_data))
                print(len(input_data[0]))
                # print(test.shape)
                print("TIME TO MAKE AN INFRENCE")
                input_data = np.array(input_data)
                input_data = np.expand_dims(input_data, axis=0)
                input_data = resize_gestures(input_data)


                input_data = torch.from_numpy(input_data).float()

                prediction = model(input_data)
                print(prediction)
                _, prediction = prediction.max(dim=1)
                print("predicted {}".format(prediction.tolist()))

                input_data = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    print("done?")
    pipeline.stop()
    cv2.destroyAllWindows()