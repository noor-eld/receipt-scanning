import cv2
import numpy as np
import pytesseract
import os


pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

imgQ = cv2.imread('../data/receipt1.jpeg')
per = 25
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ,(w//3,h//3)) # changing the display size of pop up

orb = cv2.ORB.create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
# imKp1 = cv2.drawKeypoints(imgQ, kp1, None)

path = '../data'
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(bf.match(des2, des1))
    matches.sort(key = lambda x: x.distance)
    good = matches[:int(len(matches) * (per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    # imgMatch = cv2.resize(imgMatch,(w // 3, h // 3)) # changing the display size of pop up
    cv2.imshow(y, imgMatch)

    if len(good) > 10:  # minimum matches required for stable homography
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    # imgScan = cv2.warpPerspective(img, M, (w, h))
    # cv2.imshow(y, imgScan)

        M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        if M is not None:
            imgScan = cv2.warpPerspective(img, M, (w, h))
            cv2.imshow(f"{y}-scan", imgScan)


# cv2.imshow("KeyPointsQuery",imKp1)
# cv2.imshow("Output",imgQ)
cv2.waitKey(0)