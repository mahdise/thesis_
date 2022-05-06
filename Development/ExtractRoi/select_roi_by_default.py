import cv2
import numpy as np
from Development.PipelineCamera.real_sense_camera_frame import PipelineCamera
main_object = PipelineCamera()
while True:
    frames = main_object.pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    original = color_image.copy()


    y = 30
    x = 30
    h = 250
    w = 300

    crop = original[y:y + h, x:x + w]
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (36, 255, 12), 2)


    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)

    cv2.imshow('crop part', crop)
    cv2.waitKey(1)

