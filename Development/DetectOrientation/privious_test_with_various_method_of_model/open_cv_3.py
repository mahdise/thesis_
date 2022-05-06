import cv2
import numpy as np
import math

# # read the image
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")


cap = cv2.imread("savedImage.jpg")

def nothing(x):
  pass

# create slider
cv2.namedWindow("Trackbars")
hh='Max'
hl='Min'
wnd = 'Colorbars'

cv2.createTrackbar("threshold", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("Houghlines", "Trackbars", 255, 255, nothing)


while True:
    # ret, frame = cap.read()

    frame = cv2.imread("savedImage.jpg")
    cv2.imshow("aaaaa", frame)
    key = cv2.waitKey(3)
    scale_percent = 100 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # create sliders for variables
    l_v = cv2.getTrackbarPos("threshold", "Trackbars")
    u_v = cv2.getTrackbarPos("Houghlines", "Trackbars")

    #convert frame to Black and White
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #convert Black and White to binary image
    ret,thresh4 = cv2.threshold(bw,l_v,255,cv2.THRESH_BINARY)

    #find the contours in thresh4
    contours, hierarchy = cv2.findContours(thresh4, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    #calculate with contour
    for contour in contours:

        #calculate area and moment of each contour
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)

        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        print(area)
        #Use contour if size is bigger then 1000 and smaller then 50000
        if area > 1000:
            if area <50000:
                approx = cv2.approxPolyDP(contour, 0.001*cv2.arcLength(contour, True), True)
                #draw contour
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
                #draw circle on center of contour
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                perimeter = cv2.arcLength(contour,True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                #fit elipse
                _ ,_ ,angle = cv2.fitEllipse(contour)
                P1x = cX
                P1y = cY
                length = 35
                # angle =0

                #calculate vector line at angle of bounding box
                P2x = int(P1x + length * math.cos(math.radians(angle)))
                P2y = int(P1y + length * math.sin(math.radians(angle)))
                 #draw vector line
                cv2.line(frame,(cX, cY),(P2x,P2y),(255,255,255),5)

                #output center of contour
                print  (P1x , P2y, angle)
                # a = "(" +str(P1x)+", "+str(P2y)+ ", "+ str(round(angle))+ ")"
                a = str(round(angle))
                #detect bounding box
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #draw bounding box
                cv2.drawContours(frame, [box],0,(0,0,255),2)
                cv2.putText(frame, a,  (P1x, P2y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

                #Detect Hull
                hull = cv2.convexHull(contour)
                #draw line
                #img_hull = cv2.drawContours(frame,[hull],0,(0,0,255),2)


      #print (angle)
    #    print (p)

    cv2.imshow("Frame", thresh4)
    key = cv2.waitKey(1)
    cv2.imwrite('thresh4.png', thresh4)
    key = cv2.waitKey(1)
    cv2.imshow("bw2", frame)
    key = cv2.waitKey(1)


    cv2.imwrite("12345.png",frame)
    key = cv2.waitKey(1)
    key = cv2.waitKey(1)
    #if key == 27:
     #   break
    break
    #cap.release()
    # cv2.destroyAllWindows()