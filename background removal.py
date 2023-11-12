import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting frame width and frame height as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mount_everest.jpg')

# resizing the mountain image as 640 X 480
mountain = cv2.resize(mountain, (640, 480))

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([0, 0, 0])  # Replace with lower bound RGB values
        upper_bound = np.array([255, 255, 255])  # Replace with upper bound RGB values

        # thresholding image
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask
        mask_inv = cv2.bitwise_not(mask)

        # bitwise and operation to extract foreground / person
        fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # bitwise and operation to replace background with mountain image
        bg = cv2.bitwise_and(mountain, mountain, mask=mask)
        
        # final image by combining foreground and background
        final_frame = cv2.add(fg, bg)

        # show it
        cv2.imshow('frame', final_frame)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()