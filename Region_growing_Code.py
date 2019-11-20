import math

from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random

im = Image.open('C:/Users/Harsh/Desktop/dataset/image/29030.jpg').convert('L')
out = Image.open('C:/Users/Harsh/Desktop/dataset/ground-truth/29030.png').convert('L')
arr = np.asarray(im)


rows, columns = np.shape(arr)
# print '\nrows',rows,'columns',columns
plt.figure()
plt.suptitle('Original Image')
plt.imshow(im)
plt.gray()
# User selects the intial seed point
print('\nPlease select the initial seed point')

pseed = plt.ginput(1)
# pseed
# print pseed[0][0],pseed[0][1]

x = int(pseed[0][0])
y = int(pseed[0][1])
# x = int(179)
# y = int(86)
seed_pixel = []
seed_pixel.append(x)
seed_pixel.append(y)

print('you clicked:', seed_pixel)

# # closing figure
# plt.close()

img_rg = np.zeros((rows + 1, columns + 1))
img_rg[seed_pixel[0]][seed_pixel[1]] = 255.0
img_display = np.zeros((rows, columns))

#queue
region_points = []
#add initial seed point to queue
region_points.append([x, y])


def find_region():
    print('\nloop runs till region growing is complete')
    # print 'starting points',i,j
    count = 0
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]

    while (len(region_points) > 0):

        if count == 0:
            point = region_points.pop(0)
            i = point[0]
            j = point[1]
        print('\nloop runs till queue length become zero:')
        print('len', len(region_points))
        # threshold between val-8 to val+8
        val = arr[i][j]
        lt = val - 8
        ht = val + 8
        # print 'value of pixel',val
        for k in range(8):
            # if pixel not processed
            if img_rg[i + x[k]][j + y[k]] != 1:
                try:
                    #if intensity lies within threshold range the add to queue
                    if arr[i + x[k]][j + y[k]] > lt and arr[i + x[k]][j + y[k]] < ht:
                        # print '\nbelongs to region',arr[i+x[k]][j+y[k]]
                        img_rg[i + x[k]][j + y[k]] = 1
                        p = [0, 0]
                        p[0] = i + x[k]
                        p[1] = j + y[k]
                        if p not in region_points:
                            if 0 < p[0] < rows and 0 < p[1] < columns:
                                ''' adding points to the region '''
                                region_points.append([i + x[k], j + y[k]])
                    else:
                        # print 'not part of region'
                        img_rg[i + x[k]][j + y[k]] = 0
                except IndexError:
                    continue

        # print '\npoints list',region_points
        point = region_points.pop(0)
        i = point[0]
        j = point[1]
        count = count + 1
    # find_region(point[0], point[1])


find_region()



arr_out = np.asarray(out)
ground_out = np.zeros((rows, columns))

for i in range(rows):
    for j in range(columns):
        # 0 for white and 1 for black
        if arr_out[i][j] > 125:
            ground_out[i][j] = int(0)

        else:
            ground_out[i][j] = int(1)
plt.figure()
plt.suptitle('Ground Truth')
plt.imshow(ground_out)

tp = 0
tn = 0
fn = 0
fp = 0

for i in range(rows):
    for j in range(columns):
        if ground_out[i][j] == 1 and img_rg[i][j] == 1:
            tp = tp + 1
        if ground_out[i][j] == 0 and img_rg[i][j] == 0:
            tn = tn + 1
        if ground_out[i][j] == 1 and img_rg[i][j] == 0:
            fn = fn + 1
        if ground_out[i][j] == 0 and img_rg[i][j] == 1:
            fp = fp + 1
''' ********************************** Calculation of Tpr, Fpr, F-Score, IoU ***************************************************'''

print('\n************Calculation of Tpr, Fpr, F-Score, IoU ********************')

# TP rate = TP/TP+FN
tpr = float(tp) / (tp + fn)
print("\nTPR is:", tpr)

# fp rate is
fpr = float(fp) / (fp + tn)
print("\nFPR is:", fpr)

pr = (float)(tp) / (tp + fp)
print('\n Precision:', pr)

rec = (float)(tp) / (tp + fn)
print('\n Recall:', rec)

f1 = (float)(2 * pr * rec) / (rec + pr)
print('\n F1 Score:', f1)

iou = (float)(tp) / (tp + fp + fn)
print("\nIoU Score:", iou)

plt.figure()
plt.suptitle('Segmented Image')
plt.imshow(img_rg, cmap="Greys_r")
plt.colorbar()
plt.show()