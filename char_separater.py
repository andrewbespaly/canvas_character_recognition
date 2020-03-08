import cv2
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import pandas as pd

desired_width = 170
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width, precision=3)


def separate_chars(orig_frame):

    # onedim_img = np.zeros((len(orig_frame), len(orig_frame[0])))

    # print(orig_frame[0][0])
    # if(len(orig_frame[0][0]) > 1):
    #     for y in range(0, len(orig_frame[0])):
    #         for x in range(0, len(orig_frame)):
    #             onedim_img[x][y] = orig_frame[x][y][0]
    # else:
    #     onedim_img = orig_frame

    onedim_img = orig_frame

    leftBounds = []
    rightBounds = []
    topBounds = []
    bottomBounds = []

    col_touch = False
    # col_prevTouch = False
    for col in range(0, len(onedim_img[0])):
        row_touch = False
        col_prevTouch = col_touch
        for row in range(0, len(onedim_img)):
            if onedim_img[row][col] != 0:
                if row_touch == False:
                    row_touch = True
                    col_touch = True

            if onedim_img[row][col] == 0 and row_touch == False and col_touch == True:
                col_touch = False

        if row_touch == True:
            col_touch = True

        if col_touch == True and col == len(onedim_img[0])-1:
            col_touch = False

        if col_touch == True and col_prevTouch == False:
            leftBounds.append(col)

        if col_touch == False and col_prevTouch == True:
            rightBounds.append(col)

    for char in range(0, len(leftBounds)):
        topFound = False
        for toprow in range(0, len(onedim_img)):
            for topcol in range(leftBounds[char], rightBounds[char]):
                if onedim_img[toprow][topcol] != 0 and topFound == False:
                    topFound = True
                    topBounds.append(toprow)

        botFound = False
        for botrow in range(len(onedim_img)-1, 0, -1):
            for botcol in range(leftBounds[char], rightBounds[char]):
                if onedim_img[botrow][botcol] != 0 and botFound == False:
                    botFound = True
                    bottomBounds.append(botrow)


    # print('left',leftBounds)
    # print('right',rightBounds)
    # print('top',topBounds)
    # print('bottom',bottomBounds)

    drawPix = []
    for char in range(0,len(leftBounds)):
        for u in range(leftBounds[char], rightBounds[char]):
            drawPix.append([topBounds[char], u])
            # onedim_img[topBounds[char]][u] = 255
        for r in range(topBounds[char], bottomBounds[char]):
            drawPix.append([r, rightBounds[char]])
            # onedim_img[r][rightBounds[char]] = 255
        for d in range(leftBounds[char], rightBounds[char]):
            drawPix.append([bottomBounds[char], d])
            # onedim_img[bottomBounds[char]][d] = 255
        for l in range(topBounds[char], bottomBounds[char]):
            drawPix.append([l, leftBounds[char]])
            # onedim_img[l][leftBounds[char]] = 255


    #RETURN THE POINTS UNDER ME!!!! and use like such
    # for pix in range(0, len(drawPix)):
    #     onedim_img[drawPix[pix][0]][drawPix[pix][1]] = 255

    # all_new_chars = []
    # for char in range(0, len(leftBounds)):
    #     newChar = np.zeros((bottomBounds[char]-topBounds[char], rightBounds[char]-leftBounds[char]))
    #     # for h in range(bottomBounds[char], topBounds[char]):
    #     for h in range(topBounds[char], bottomBounds[char]):
    #         for w in range(leftBounds[char], rightBounds[char]):
    #             newChar[h-bottomBounds[char]][w-leftBounds[char]] = onedim_img[h][w]
    #     all_new_chars.append(newChar)

    borderPad = 25
    for char in range(0, len(leftBounds)):
        wide = rightBounds[char] - leftBounds[char]
        tall = bottomBounds[char] - topBounds[char]

        if wide >= tall:
            canvLength = wide
            borderstart = (canvLength - tall) / 2
        else:
            canvLength = tall
            borderstart = (canvLength - wide) / 2
        borderstartpad = borderPad / 2

        newCanvas = np.zeros((tall, wide))

        for canvY in range(0, tall):
            for canvX in range(0, wide):
                newCanvas[canvY][canvX] = onedim_img[topBounds[char]+canvY][leftBounds[char]+canvX]

        for b in range(0, round(borderstart+borderstartpad)):
            newCanvas = np.insert(newCanvas, 0, 0, axis=1)
            newCanvas = np.insert(newCanvas, len(newCanvas[0]), 0, axis=1)
        for b in range(0, round(borderstartpad)):
            newCanvas = np.insert(newCanvas, 0, 0, axis=0)
            newCanvas = np.insert(newCanvas, len(newCanvas), 0, axis=0)

        smallCanvas = cv2.resize(newCanvas, (28, 28), interpolation=cv2.INTER_AREA)
        smallCanvas = np.round(smallCanvas)

        # cv2.imshow('t', newCanvas)
        # cv2.imshow('s', smallCanvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        smallCanvas = smallCanvas.flatten()
        smallCanvas = np.true_divide(smallCanvas, 255)
        print(smallCanvas)

        with tf.Session() as sess:

            saver = tf.train.import_meta_graph('mnist_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            feed_dict = {x: [smallCanvas]}
            y_ = graph.get_tensor_by_name("y_:0")
            print(sess.run(tf.math.argmax(y_[0]), feed_dict))




if __name__ == '__main__':
    goodtest = cv2.imread(r'chartest.png')
    # cv2.imshow('onedim', goodtest)
    separate_chars(goodtest)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



