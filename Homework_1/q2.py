import cv2
import numpy as np  
import matplotlib.pyplot as plt

img_count = 0
def histogram(source_image):
    global img_count
    plt.hist(source_image.ravel(), 256, (0, 256))
    plt.savefig("output/historgram_" + str(img_count) + ".png")
    print("Histogram saved in historgram_" + str(img_count) + ".png")
    plt.clf()
    img_count += 1

img = cv2.imread('images/grayscale_1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/grayscale_2.jpg', cv2.IMREAD_GRAYSCALE)
histogram(img)
histogram(img2)
