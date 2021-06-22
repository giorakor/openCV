from typing import Counter
import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
cv2.namedWindow("res")
cv2.namedWindow("mask")
img_counter = 0
last = time.time()

lower = np.array([35, 40, 50])
upper = np.array([75, 255, 200])
min_size = 1000

while True:
    ret, frame = cam.read()
    start = time.time()
    if not ret:
        print("failed to grab frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cv2.imshow("mask", mask)

    nb_components, all_blobs, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    sizes = stats[1:, -1]
    for i in range(0, nb_components-1):  # loop through all blobs
        if sizes[i] > min_size:
            blob_i = cv2.inRange(all_blobs, i + 1, i + 1)
            contours, hierarchy = cv2.findContours(blob_i, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            for j in range(0, len(contours)):  # loop through all contours in a blob
                cnt = contours[j]
                cv2.drawContours(frame, [cnt], 0, (250, 0, 250), 2)
                if hierarchy[0, j, 3] == -1:  # no parent contour - i.e - the outer contour only
                    area = cv2.contourArea(cnt)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area
                    if solidity > 0.3 and solidity < 0.5:
                        poly_approx = cv2.approxPolyDP(cnt,
                                                       0.018 * cv2.arcLength(cnt, True), True)
                        if(len(poly_approx) == 8):
                            cv2.drawContours(
                                frame, [poly_approx], 0, (0, 255, 255), 2)
                            for pnt in poly_approx:
                                cv2.circle(frame, pnt[0], 5, (255, 0, 0), 2)

    #print(time.time() - start, " ", time.time()-last)
    last = time.time()
    cv2.imshow("res", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
