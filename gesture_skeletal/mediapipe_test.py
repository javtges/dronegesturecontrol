import mediapipe as mp
import cv2


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def find_hand(frame):
        with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.35, min_tracking_confidence=0.35, max_num_hands=1) as hands:
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
               

camera = cv2.VideoCapture("/dev/video0")

while(True):

    ret, frame = camera.read()

    find_hand(frame)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
camera.release()
# Destroy all the windows
cv2.destroyAllWindows()