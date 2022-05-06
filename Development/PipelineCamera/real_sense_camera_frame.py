import cv2
import numpy as np
import pyrealsense2 as rs


class PipelineCamera:
    def __init__(self, chosen_camera_serial_number, camera_res=(1280, 720)):

        self.break_loop = False
        self.frames = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if chosen_camera_serial_number:
            self.config.enable_device(chosen_camera_serial_number)

        self.config.enable_stream(rs.stream.depth, camera_res[0], camera_res[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, camera_res[0], camera_res[1], rs.format.bgr8, 30)

        self.pipeline.start(self.config)

    def get_image(self, image_type="color_image"):

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                if image_type == "color_image":
                    print(color_image)
                    return color_image
                elif image_type == "depth_image":
                    return depth_image

                elif image_type == "both":
                    return color_image, depth_image

        finally:

            # Stop streaming
            self.pipeline.stop()


if __name__ == '__main__':
    # ['841512070160', '739112060148']
    main_object = PipelineCamera(chosen_camera_serial_number="739112060148")
    x = 1
    while True:
        frames = main_object.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
