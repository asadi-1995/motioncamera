import os
import time
from datetime import datetime

import cv2

CAMERA_INDEXT = 0     
MIN_AREA = 1500   
COOLDOWN_SECONDS = 2.0 
SAVE_DIR = "captures"

def main():
    os.makedirs(SAVE_DIR,exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEXT)
    
    if not cap.isOpened:
        print("can not open the camera")
        return

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500 , varThreshold=50 , detectShadows=False
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    last_saved = 0.0
    frame_count = 0

    #let the background settle model for a moment
    warmup_frames = 30
    print("running press (q) to quit")

    while True:
        ok , frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame,1)

        small = cv2.resize(frame,(640,int(frame.shape[0] * 640 / frame.shape[1])))

        mask = subtractor.apply(small)
        mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask,kernel,iterations=2)

        frame_count += 1
        motion = False

        if frame_count > warmup_frames :
            contours = cv2.findContours(
                mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2]
            for c in contours:
                if cv2.contourArea(c) < MIN_AREA:
                    continue
                motion = True
                x , y , w , h = cv2.boundingRect(c)
                cv2.rectangle(small , (x , y) , (x + w, y + h) , (0 , 255 , 0), 2)
            now = time.time()
            if motion and now - last_saved >= COOLDOWN_SECONDS:
                name = datetime.now().strftime("%y%m%d_%H%M%S")+".jpg"
                cv2.imwrite(os.path.join(SAVE_DIR,name),frame)
                last_saved = now
                print("saved",name)
        cv2.imshow("camera",small)
        cv2.imshow("mask" , mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
