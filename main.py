import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#__Audio setup__#
devices = AudioUtilities.GetDeviceEnumerator()
device = devices.GetDefaultAudioEndpoint(0, 1)

interface = device.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# __Camera__#
cap = cv2.VideoCapture(0)

#__Hand Tracking__#
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

mpDraw = mp.solutions.drawing_utils

#__Smoothing__#
prevVol = 0
prevBrightness = 0
smoothness = 0.2

# Stable values
lastVolPer = 0
lastBrightPer = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hand_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:

        hand_detected = True

        for idx, handLms in enumerate(results.multi_hand_landmarks):

            label = results.multi_handedness[idx].classification[0].label

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            length = math.hypot(x2 - x1, y2 - y1)
            length = np.clip(length, 30, 200)

            #__Volume(Left Hand)__#
            if label == "Left":
                vol = np.interp(length, [30, 200], [volMin, volMax])
                vol = prevVol * (1 - smoothness) + vol * smoothness
                prevVol = vol

                volume.SetMasterVolumeLevel(vol, None)
                lastVolPer = np.interp(length, [30, 200], [0, 100])

            #__Brightness(Right Hand)__#
            elif label == "Right":
                brightness = np.interp(length, [30, 200], [0, 100])

                if abs(brightness - prevBrightness) > 2:
                    sbc.set_brightness(int(brightness))
                    prevBrightness = brightness

                lastBrightPer = brightness

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Keep last values (no reset)
    if not hand_detected:
        lastVolPer = lastVolPer
        lastBrightPer = lastBrightPer

    #__HUD(Top left corner)__#

    hud_x, hud_y = 20, 20
    hud_w, hud_h = 200, 120

    overlay = img.copy()
    cv2.rectangle(overlay, (hud_x, hud_y),
                  (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)

    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    cv2.rectangle(img, (hud_x, hud_y),
                  (hud_x + hud_w, hud_y + hud_h), (255, 255, 255), 1)

    cv2.putText(img, "A.Ges.Co.S", (hud_x + 10, hud_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(img, f"Vol: {int(lastVolPer)}%", (hud_x + 10, hud_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(img, f"Bright: {int(lastBrightPer)}%", (hud_x + 10, hud_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("A.Ges.Co.S - Air Gesture Control System", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
