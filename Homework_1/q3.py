import cv2
import numpy as np  
import matplotlib.pyplot as plt

def otsu_threshold(source_image):
    (h_img, w_img) = source_image.shape
    pixel_number = h_img * w_img
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(source_image, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        sumb = np.sum(his[:t])
        sumf = np.sum(his[t:])
        Wb = sumb * mean_weight
        Wf = sumf * mean_weight

        if float(sumb)== 0 or float(sumf)==0:
            continue

        mb = np.sum(intensity_arr[:t]*his[:t]) / float(sumb) #mean background
        mf = np.sum(intensity_arr[t:]*his[t:]) / float(sumf) #mean foreground

        value = Wb * Wf * (mb - mf) ** 2 # between class variance

        if value > final_value:
            final_thresh = t
            final_value = value

    res_img = source_image.copy()
    res_img[source_image > final_thresh] = 255
    res_img[source_image < final_thresh] = 0
    return res_img

img = cv2.imread('images/otsu_1.jpg', cv2.IMREAD_GRAYSCALE)
binary_image = otsu_threshold(img)
cv2.imwrite('output/otsu_1.jpg', binary_image)

img2 = cv2.imread('images/otsu_2.png', cv2.IMREAD_GRAYSCALE)
binary_image2 = otsu_threshold(img2)
cv2.imwrite('output/otsu_2.jpg', binary_image2)
