import cv2

# main_object = Pipeline(choosen_camera="841512070160")
refPt = []
cropping = False
crop = None


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False


# while True:
#     frames = main_object.pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
#     if not depth_frame or not color_frame:
#         continue
#
#     # Convert images to numpy arrays
#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#
#     clone = color_image.copy()
#     # cv2.namedWindow("image")
#     cv2.setMouseCallback("RealSense", click_and_crop)
#
#     default = True
#     if len(refPt) == 2:
#         if refPt[0] != refPt[1]:
#             y = refPt[0][1]
#             x = refPt[0][0]
#             y_h = refPt[1][1]
#             x_w = refPt[1][0]
#
#             crop = clone[y:y_h, x:x_w]
#             cv2.rectangle(color_image, refPt[0], refPt[1], (36, 255, 12), 2)
#             default = False
#         else:
#             default = True
#
#     if default:
#         y = 30
#         x = 30
#         h = 250
#         w = 300
#
#         crop = clone[y:y + h, x:x + w]
#         cv2.rectangle(color_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
#
#     # here op will only in the crop image
#     if crop is not None:
#         # DetectOrientation(crop)
#         detect_arrow_direction = DetectArrowWithCanny(crop)
#
#         # del detect_arrow_direction
#     # Show images
#     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('RealSense', color_image)
#     if crop is not None:
#         cv2.imshow('ROI', crop)
#
#     # cv2.imshow('crop part', crop)
#     cv2.waitKey(1)
