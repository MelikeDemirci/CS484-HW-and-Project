import numpy as np   
import cv2

def checkNeighbours(h,w,img,se,mode = 0):
    (h_img, w_img) = img.shape
    (h_se, w_se) = se.shape

    #Find the edge indexes for the slices
    left = int(w - (w_se - 1) / 2)
    if left < 0:
        left = 0
    right = int(w + (w_se - 1) / 2)
    if right >= w_img:
        right = w_img - 1
    top = int(h - (h_se - 1) / 2)
    if top < 0:
        top = 0
    bottom = int(h + (h_se - 1) / 2)
    if bottom >= h_img:
        bottom = h_img - 1

    #Slice the image with respect to the shape of structuring element
    sliced = img[top:bottom+ 1, left : right + 1] 
    if mode == 0:
        return len(set(sliced.flatten()).intersection(set(se.flatten()))) >= 1
    elif mode == 1:
        return list(sliced.flatten()) == list(se.flatten())

def dilation(img,se):
    (h_img, w_img) = img.shape
    res_img = np.zeros(img.shape).astype(np.uint8)

    for h in range(h_img):
            for w in range(w_img):
                if checkNeighbours(h,w,img,se, 0):
                    res_img[h][w] = 255
    return res_img

def erosion(img,se):
    (h_img, w_img) = img.shape
    res_img = np.zeros(img.shape).astype(np.uint8)

    for h in range(h_img):
            for w in range(w_img):
                if checkNeighbours(h,w,img,se, 1):
                    res_img[h][w] = 255
    return res_img

src_img = cv2.imread('images/binary_image.png', cv2.IMREAD_GRAYSCALE)
struct_et = np.ones((2, 2)) * 255
eroded = erosion(src_img, struct_et)
struct_et = np.ones((4, 4)) * 255
clean_image = dilation(eroded, struct_et)
cv2.imwrite('output/clean.png', clean_image)
