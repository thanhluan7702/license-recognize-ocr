import cv2
import imutils
import math
import numpy as np 

image = cv2.imread('GreenParking/0513_00490_b.jpg')

image = imutils.resize(image, width=300)
cv2.imshow("original image", image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyed image", gray_image)

gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17) 
cv2.imshow("smoothened image", gray_image)

edged = cv2.Canny(gray_image, 30, 200) 
cv2.imshow("edged image", edged)

cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("contours",image1)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

screenCnt = []

image2 = image.copy()
cv2.drawContours(image2,cnts,-1,(0,255,0),3)
cv2.imshow("Top 5 contours",image2)

for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

    area = cv2.contourArea(c)

    if len(approx) == 4: 
        screenCnt.append(approx)
        print(screenCnt)

        x,y,w,h = cv2.boundingRect(c) 
        new_img=image[y:y+h,x:x+w]
        # cv2.imwrite('./'+str(i)+'.png',new_img)
        cv2.imshow("License Plate", new_img)
        break

if screenCnt == []:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        ############## Find the angle of the license plate #####################
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

                ####################################

        ########## Crop out the license plate and align it to the right angle ################

        mask = np.zeros(gray_image.shape, np.uint8)

        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

        # Cropping
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = image[topx:bottomx, topy:bottomy]

        imgThresh = gray_image[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        print(rotationMatrix)
        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))

        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        cv2.imshow("roi", roi)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
        cv2.imshow("imgThresh", imgThresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

