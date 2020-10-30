# @Author: ASHISH SASMAL <ashish>
# @Date:   30-10-2020
# @Last modified by:   ashish
# @Last modified time: 30-10-2020

import cv2
import sys
import numpy as np

img1 = cv2.imread(f"Test/{sys.argv[1]}")
# face_cascade = cv2.CascadeClassifier('face_detector.xml')
#
# faces = face_cascade.detectMultiScale(img, 1.1, 4)
#
# for (x, y, w, h) in faces:
#   cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
import time

start_time = time.time()
import face_recognition as fg

img = fg.load_image_file(f"Test/{sys.argv[1]}")
locs = fg.face_locations(img)

# print(locs,len(locs))
print("--- %s seconds ---" % (time.time() - start_time))


known_image = fg.load_image_file(f"Test/{sys.argv[2]}")
enc1 = fg.face_encodings(known_image)[0]

match_loc = []

for x,y,w,h in locs:

    sam = img1[x:w, h:y]

    enc2 = fg.face_encodings(sam)
    if enc2:
        enc2=enc2[0]

        results = fg.compare_faces([enc1], enc2)[0]
        if results:
            match_loc.append((x,y,w,h))

x,y,w,h = match_loc[0]
# cv2.rectangle(img1, (h,x), (y, w), (255, 0, 0), 2)

face_blur = img1[x:w, h:y]
kernel = np.ones((10,10),np.float32)/100
img1[x:w, h:y] = cv2.filter2D(face_blur,-1,kernel)

# cv2.imwrite("Result/face_detected.png",img1)
cv2.imshow("Result",img1)
cv2.waitKey(0)
