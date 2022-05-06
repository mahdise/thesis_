import cv2
import numpy as np

cv2.useOptimized()


def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x,y,disp[y,x],filteredImg[y,x]
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        Distance = -593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance: ' + str(Distance) + ' m')


cv2.useOptimized()

# Parameters for Distortion Calibration

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0, 50):  # Put the amount of pictures you have taken for the calibration
    t = str(i)
    ChessImaR = cv2.imread('images/chessboard-R' + t + '.png', 0)  # Right side
    ChessImaL = cv2.imread('images/chessboard-L' + t + '.png', 0)  # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9, 6), None)  # Define the number of chees corners
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9, 6), None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1], None, None)
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                            (wR, hR), 1, (wR, hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1], None, None)
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))
cv2.useOptimized()

print('Cameras Ready to use')

# Calibrate the Cameras for Stereo

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                           imgpointsL,
                                                           imgpointsR,
                                                           mtxL,
                                                           distL,
                                                           mtxR,
                                                           distR,
                                                           ChessImaR.shape[::-1],
                                                           criteria_stereo,
                                                           flags)

# StereoRectify function
rectify_scale = 1  # 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                  ChessImaR.shape[::-1], R, T,
                                                  rectify_scale,
                                                  (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1],
                                              cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)

cv2.useOptimized()

# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 16  # 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=0,
                               speckleWindowSize=10,
                               speckleRange=2,
                               disp12MaxDiff=5,
                               preFilterCap=5,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
cv2.useOptimized()

# Starting the StereoVision

# Call the two cameras
CamR = cv2.VideoCapture(1)  # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL = cv2.VideoCapture(2)

while True:
    # Start Reading Camera images
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    cv2.imshow('CAMR', frameR)
    cv2.imshow('CAML', frameL)

    # Rectify the images on rotation and alignement
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                          0)  # Rectify the image using the kalibration parameters founds during the initialisation
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Convert from color(BGR) to gray
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

    # 'Output after Rectifying RightCam & LeftCam'
    cv2.imshow('GrayR', grayR)
    cv2.imshow('GrayL', grayL)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
    dispL = disp
    dispR = stereoR.compute(grayR, grayL)

    dispL = np.int16(dispL)
    dispR = np.int16(dispR)
    cv2.useOptimized()

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp

    # Colors map
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    # Show the result for the Depth_image
    cv2.imshow('Filtered Color Depth', filt_Color)

    # Mouse click
    cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, filt_Color)

    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the Cameras
CamR.release()
CamL.release()
