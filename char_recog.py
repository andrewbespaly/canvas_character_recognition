import idx2numpy
import pandas as pd
import numpy as np
import imutils
import time
import cv2
import colorsys
import keyboard

import char_separater



# FLAGS TO VISUALIZE TRACKING
SHOW_MASK = True
SHOW_CANVAS = True
SHOW_PRODUCT = True

frame_size = (640, 480)
default_thickness = 10
default_color = (0,0,255)  # Red in BGR

####################################################
desired_width = 170
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width, precision=3)
# lab_file = r'emnist-balanced-train-labels-idx1-ubyte'
# img_file = r'emnist-balanced-train-images-idx3-ubyte'
# labels = idx2numpy.convert_from_file(lab_file)
# imgs = idx2numpy.convert_from_file(img_file)
# num = 4
# for y in range(0, len(imgs[0])):
#     for x in range(0, len(imgs[0][0])):
#         print(imgs[num][x][y], end='\t')
#     print()
# print(labels[num])
###################################################

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = frame[y,x,0]
        colorsG = frame[y,x,1]
        colorsR = frame[y,x,2]
        colors = frame[y,x]
        # print("Red: ",colorsR)
        # print("Green: ",colorsG)
        # print("Blue: ",colorsB)
        # print("BRG Format: ",colors)
        hsvconv = colorsys.rgb_to_hsv(colorsR, colorsG, colorsB)
        hsvnew = [hsvconv[0] * 180, hsvconv[1] * 255, hsvconv[2]]
        # print('HSV Format: ',hsvnew)
        newRGB = colorsys.hsv_to_rgb(hsvnew[0]/180, hsvnew[1]/255, hsvnew[2])
        # print('UPDATED RGB: ', newRGB)
        # print("Coordinates of pixel: X: ",x,"Y: ",y)
        global bgr_poten_track_color
        bgr_poten_track_color = colors
        global hsv_poten_track_color
        hsv_poten_track_color = hsvnew


cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouseRGB)

vs = cv2.VideoCapture(0)


time.sleep(2.0)

hsv_poten_track_color = None
bgr_poten_track_color = None
track_color = None

colorSearch = False
trackingLive = False

canvas = np.zeros((frame_size[1], frame_size[0]))
prevcenter = []
points = []
quit = False
drawing = False
wasdrawing = False
ready_to_submit = False

fileInfo = char_separater.read_data_file('emnist-balanced-mapping.txt')


# Main loop
while quit == False:
    # grab the current frame
    frame = vs.read()

    frame = frame[1]

    # in case error occurred
    if frame is None:
        break

    # flip frame of webcam for ease of user
    frame = np.fliplr(frame)

    # resize the frame
    frame = cv2.resize(frame, dsize=frame_size)

    # display canvas if desired
    if SHOW_CANVAS == True:
        cv2.imshow('canvas', canvas)


    # if(trackingLive == True):
    #     frame = np.where(canvas != 0, frame, (0,0,255))

    # directions when first opened
    if(colorSearch == False and trackingLive == False):
        frame = cv2.putText(frame, 'Click a color you want to track', (90, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),1)

    # set new color to track
    if np.any(bgr_poten_track_color != None):
        if(trackingLive == False):
            colorSearch = True
            color_sqr = (int(bgr_poten_track_color[0]), int(bgr_poten_track_color[1]), int(bgr_poten_track_color[2]))
            frame = cv2.rectangle(frame, (20,20), (40,40), color_sqr, -1)
            cv2.putText(frame, 'Enter To Confirm Tracking Color', (50,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10),1)

        # track color on screen
    if trackingLive == True:
        colorSearch = False
        trackingLive = True

        # define lower and upper bounds of color to track in hsv colorspace
        lowerColorBound = [track_color[0]-20, track_color[1]-50, track_color[2]-50]
        upperColorBound = [track_color[0]+20, track_color[1]+100, track_color[2]+100]

        # set boundaries in case the color is close to edge
        for i in range(0,3):
            if(lowerColorBound[i] < 0):
                lowerColorBound[i] = 0
            # hsv colorspace, h goes up to 180
            if(i == 0):
                if(lowerColorBound[i] > 180):
                    lowerColorBound[i] = 180
            else:
                if (lowerColorBound[i] > 255):
                    lowerColorBound[i] = 255

            if (upperColorBound[i] < 0):
                upperColorBound[i] = 0

            # hsv colorspace, h goes up to 180
            if (i == 0):
                if (upperColorBound[i] > 180):
                    upperColorBound[i] = 180
            else:
                if (upperColorBound[i] > 255):
                    upperColorBound[i] = 255

        lowerColor = (int(lowerColorBound[0]), int(lowerColorBound[1]), int(lowerColorBound[2]))
        upperColor = (int(upperColorBound[0]), int(upperColorBound[1]), int(upperColorBound[2]))

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


        if (drawing == True and wasdrawing == False):
            drawing = False
            wasdrawing = True

        for i in range(0, len(points)-1):
            if np.all(points[i] >= (0,0)) and np.all(points[i+1] >= (0,0)):
                cv2.line(frame, points[i], points[i+1], default_color, default_thickness)
            else:
                cv2.line(frame, points[i+1], points[i+1], default_color, default_thickness)


        mask = cv2.inRange(hsv, lowerColor, upperColor)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=5)
        if SHOW_MASK == True:
            cv2.imshow('mask', mask)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None


        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            prevcenter.append(center)
            if(len(prevcenter) > 2):
                prevcenter.pop(0)

            # only proceed if the radius meets a minimum size
            if radius > 3:
                # draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 1)
                cv2.circle(frame, center, default_thickness, (0, 0, 255), -1)



    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if keyboard.is_pressed('q'):
        quit = True

    # Enter to confirm color
    elif keyboard.is_pressed('enter') and colorSearch == True:
        track_color = hsv_poten_track_color
        trackingLive = True

    # 'r' to change to another color to track
    elif keyboard.is_pressed('r') and trackingLive == True:
        trackingLive = False
        colorSearch = True
    # 'e' to clear all the drawings
    elif keyboard.is_pressed('e') and trackingLive == True:
        canvas = np.zeros_like(canvas)
        points.clear()
        ready_to_submit = False

    # space to draw
    elif keyboard.is_pressed(' ') and trackingLive == True:
        ready_to_submit = True
        cv2.line(canvas, prevcenter[-2], prevcenter[-1], 255, default_thickness)
        points.append(prevcenter[-1])
        drawing = True
        wasdrawing = False

    elif keyboard.is_pressed('up'):
        if default_thickness < 30:
            default_thickness += 1

    elif keyboard.is_pressed('down'):
        if default_thickness > 1:
            default_thickness -= 1

    elif wasdrawing == True:
        points.append((-100,-100))
        wasdrawing = False

    # 'Enter' while in drawing stage to turn in your letter/number and get result
    elif keyboard.is_pressed('enter') and trackingLive == True and ready_to_submit == True:
        resultList = char_separater.separate_chars(canvas, fileInfo)
        print(resultList)









cv2.destroyAllWindows()
