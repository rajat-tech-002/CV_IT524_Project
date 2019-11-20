import cv2
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

# Definition for a QuadTree node.
class Point(object):
    def __init__(self , x , y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y

class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight, location):
        self.mean_val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
        self.location = location


class Solution:
    def __init__(self, image, total_mean, total_max, total_min):
        self.image = image
        self.total_mean = total_mean
        self.total_max = total_max
        self.total_min = total_min

    def construct(self, picture, location):

        root = Node(None, False, None, None, None, None, location)

        if picture.shape[0] == 1: #if leaf node
            root.isLeaf = True
            root.mean_val = picture.mean()
            self.color(location, root.mean_val)

        elif self.stop_split(picture):  # Determine whether to continue dividing
            root.isLeaf = True
            root.mean_val = picture.mean()
            self.color(location, root.mean_val)

        else:  # Continue to split
            height = picture.shape[0]
            width = picture.shape[1]
            halfheight = height // 2
            halfwidth = width // 2 # Use // to mean divisibility
            root.isLeaf = False  #If the values ​​in the grid are not equal, this node is not a leaf node.
            #print(height, width, halfheight, halfwidth)

            # Autoregressive
            # Base is the submap reference
            base_start_x = location[0].x
            base_start_y = location[0].y
            base_end_x = location[1].x
            base_end_y = location[1].y

            #quadrants
            root.topLeft = self.construct(picture[:halfheight, :halfwidth], [Point(base_start_x, base_start_y), Point(base_start_x + halfheight, base_start_y + halfwidth)])
            root.topRight = self.construct(picture[:halfheight, halfwidth:], [Point(base_start_x, base_start_y + halfwidth), Point(base_start_x + halfheight, base_end_y)])
            root.bottomLeft = self.construct(picture[halfheight:, :halfwidth], [Point(base_start_x + halfheight, base_start_y), Point(base_end_x, base_start_y + halfwidth)])
            root.bottomRight = self.construct(picture[halfheight:, halfwidth:], [Point(base_start_x + halfheight, base_start_y + halfwidth), Point(base_end_x, base_end_y)])
        return root

    def stop_split(self, iim):
        #stopping criteria
        if iim.max() - iim.min() <= 10:
            return True
        else:
            return False

    def color(self, location, mean_val):
        first_point = location[0]
        second_point = location[1]
        # print(first_point.x, first_point.y, ' to ', second_point.x, second_point.y)
        if mean_val <= self.total_mean:
            for i in range(first_point.x, second_point.x):
                for j in range(first_point.y, second_point.y):
                    self.image[i][j] = self.total_max
        else:
            for i in range(first_point.x, second_point.x):
                for j in range(first_point.y, second_point.y):
                    self.image[i][j] = self.total_min

if __name__ == '__main__':

    # import Image
    im = cv2.imread('C:/Users/Harsh/Desktop/dataset/image/29030.jpg', 0)
    out = Image.open('C:/Users/Harsh/Desktop/dataset/ground-truth/29030.png').convert('L')
    arr_out = np.asarray(out)

    im_shape = im.shape
    height = im_shape[0]
    width = im_shape[1]

    plt.figure()
    plt.suptitle('Original Image')
    plt.imshow(im)
    plt.gray()
    print('the shape of image :', im_shape)
    # new_name = name[:-4] + '_treated.jpg'

    init_locate = [Point(0, 0), Point(height, width)] # Diagonal two-point positioning rectangle
    # Zpd object of class solution
    zpd = Solution(im, im.mean(), im.max(), im.min())
    # call construct method of solution class
    zpd.construct(im, init_locate)

    #convert result to binary image
    result=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            # 0 for black and 255 for white
            if zpd.image[i][j] > 125:
                result[i][j] = int(1)

            else:
                result[i][j] = int(0)

    #convert ground truth to binary image
    ground_out = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            # 0 for black and 1 for white
            if arr_out[i][j] > 125:
                #convert white pixels to black
                ground_out[i][j] = int(0)

            else:
                #convert black pixels to white
                ground_out[i][j] = int(1)
    plt.figure()
    plt.suptitle('Ground Truth')
    plt.imshow(ground_out)

    tp = 0
    tn = 0
    fn = 0
    fp = 0

    for i in range(height):
        for j in range(width):
            if ground_out[i][j] == 1 and result[i][j] == 1:
                tp = tp + 1
            if ground_out[i][j] == 0 and result[i][j] == 0:
                tn = tn + 1
            if ground_out[i][j] == 1 and result[i][j] == 0:
                fn = fn + 1
            if ground_out[i][j] == 0 and result[i][j] == 1:
                fp = fp + 1
    ''' ********************************** Evaluation***************************************************'''

    print('\n************Calculation of Precision,Recall, F1-Score, IoU ********************')

    # TP rate
    tpr = float(tp) / (tp + fn)
    print("\nTPR is:", tpr)

    # FP rate
    fpr = float(fp) / (fp + tn)
    print("\nFPR is:", fpr)

    pr=(float)(tp)/(tp+fp)
    print('\n Precision:',pr)

    rec=(float)(tp)/(tp+fn)
    print('\n Recall:',rec)

    f1=(float)(2*pr*rec)/(rec+pr)
    print('\n F1 Score:', f1)

    iou = (float)(tp) / (tp + fp + fn)
    print("\nIoU Score:", iou)

    plt.figure()
    plt.suptitle('Segmentated Image')
    plt.imshow(result)
    plt.colorbar()
    plt.show()

    # while True:
    #     cv2.imshow('OUTIMAGE' , zpd.image)
    #     c = cv2.waitKey(30) & 0xff
    #     if c == 27:
    #         cv2.imwrite(new_name, zpd.image)
    #         break