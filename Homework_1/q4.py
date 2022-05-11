import numpy as np   
import cv2

def convolution2d(source_image, kernel):
    (h_img, w_img)  = source_image.shape
    (h_kernel, w_kernel) = kernel.shape
    
    H = (h_kernel - 1) // 2
    W = (w_kernel - 1) // 2
    
    res_img = np.zeros((h_img, w_img))

    for i in np.arange(H, h_img-H):
        for j in np.arange(W, w_img-W):
            sum = 0
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    p = source_image[i+k, j+l]
                    r = kernel[H+k, W+l]
                    sum += (p * r)
            res_img[i,j] = sum 
    return res_img

src_img = cv2.imread('images/filter.jpg', cv2.IMREAD_GRAYSCALE)

#-----------------Sobel Operation------------------------
sob_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
sob_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

# Normalization
sobel_x = convolution2d(src_img, sob_x)/8.0
sobel_y = convolution2d(src_img, sob_y)/8.0

sobel_out = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
sobel_out = (sobel_out / np.max(sobel_out)) * 255
cv2.imwrite('output/sobel.jpg', sobel_out)

#-----------------Prewitt Operation------------------------
prew_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
prew_y = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])

# Normalization
prewitt_x = convolution2d(src_img, prew_x)/8.0
prewitt_y = convolution2d(src_img, prew_y)/8.0

prewitt_out = np.sqrt(np.power(prewitt_x, 2) + np.power(prewitt_y, 2))
prewitt_out = (prewitt_out / np.max(prewitt_out)) * 255
cv2.imwrite('output/prewitt.jpg', prewitt_out)