import cv2
import numpy as np
from matplotlib import pyplot as plt



class DetectArrowWithCanny:
    def __init__(self, image):
        # print(len(image))

        result = ".............Not yet........."
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        for i in corners:
            x, y = i.ravel()
            print(x,y)
            cv2.circle(img, (x, y), 3, 255, -1)
        plt.imshow(img), plt.show()

        print("..........................")

        edges = cv2.Canny(gray, 100, 200)
        init_contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in init_contours]

        angee_list = {}
        points = [(x[0][0],x[0][1]) for cnt in contours for x in cnt]
        print(points)
        for i in range(len(points)-1):
            first_tup = np.matrix(points[i])
            second_tup = np.matrix(points[i+1])
            my_angle = (np.arctan(second_tup-first_tup)*180/np.pi)
            angee_list[i] = my_angle
            print(my_angle)

        print(len(angee_list))
        print(angee_list)

        plt.subplot(122)
        plt.imshow(edges, cmap="gray")
        plt.show()





if __name__ == '__main__':
    img = cv2.imread(
        "/Development/priviuos_test/detect_arrow/arrowtest4.png")

    detect_arrow = DetectArrowWithCanny(img)