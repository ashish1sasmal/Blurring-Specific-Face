# @Author: ASHISH SASMAL <ashish>
# @Date:   31-10-2020
# @Last modified by:   ashish
# @Last modified time: 31-10-2020

import cv2
import sys
import numpy as np
import time
import face_recognition as fg

known_image = fg.load_image_file(f"Test/{sys.argv[2]}")
face_cascade = cv2.CascadeClassifier('face_detector.xml')
enc1 = fg.face_encodings(known_image)[0]


vid = cv2.VideoCapture(f"Test/{sys.argv[1]}")



while True:
    img1 = vid.read()[1]
    match_loc = []
    locs = face_cascade.detectMultiScale(img1, 1.1, 4)
    for x,y,w,h in locs:
        sam = img1[y:y+h,x:x+w]
        enc2 = fg.face_encodings(sam)
        if enc2:
            enc2=enc2[0]
            results = fg.compare_faces([enc1], enc2)[0]
            if results:
                match_loc.append((x,y,w,h))
    # print(match_loc)
    if match_loc:
        x,y,w,h = match_loc[0]
        face_blur = img1[y:y+h, x:x+w]
        kernel = np.ones((10,10),np.float32)/100
        img1[y:y+h, x:x+w] = cv2.filter2D(face_blur,-1,kernel)

    cv2.imshow("Result",img1)
    k = cv2.waitKey(1)
    if k==27:
        break

vid.release()
cv2.destroyAllWindows()
