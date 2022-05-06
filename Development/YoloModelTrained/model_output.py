import numpy as np
import cv2

from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont

def draw_outputs(img, outputs, class_names):
    copy_image = img
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='./data/fonts/futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)

    for i in range(nums):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=tuple(color))

        confidence = '{:.2f}%'.format(objectness[i] * 100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                       fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black',
                  font=font)

        ####################################################################
        # __calculate_arrow(copy_image,x1y1, x2y2)
        # __get_angle(copy_image,x1y1, x2y2)
        #
        # # find contours in the edge map
        # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # for c in cnts:
        #     # compute the rotated bounding box of the contour
        #     box = cv2.minAreaRect(c)
        #     box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        #     box = np.array(box, dtype="int")

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # # Convert image to binary
        # _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img, contours, i, (0, 0, 255), 2)

        # crop = a[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0] ]
        #
        # crop_list[i]=a[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0] ]

        # crop = a[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]

    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)

    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img