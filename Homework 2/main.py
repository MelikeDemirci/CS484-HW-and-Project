# MELİKE DEMİRCİ - 21702346
import cv2
import numpy as np

drawCount = 0
imgWidth = 0
def load_images(filename):
    images = []
    files = []

    with open(filename, 'r') as file:
        for line in file:
            files.append(line.replace('\n', ''))

    for file in files:
        img = cv2.imread('data/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
 
    return images

def match(d1, d2):
    matcher = cv2.BFMatcher(cv2.NORM_L1)
    raw_matches = matcher.knnMatch(d1, d2, k=2)
    good_points = []
    good_matches= []
    for m1, m2 in raw_matches:
        if m1.distance < 0.70 * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])
    
    return good_points, good_matches

def blend(warped, img1, img2):
    windowWidth = 30
    im1 = img1[:, img1.shape[1]-windowWidth:]
    im2 = warped[:, img1.shape[1]-windowWidth:img1.shape[1]]
    warped[:img1.shape[0],:img1.shape[1]-windowWidth] = img1[:img1.shape[0],:img1.shape[1]-windowWidth]
    avg = im1 * 0.5 + im2 * 0.5
    warped[: , img1.shape[1]-windowWidth:img1.shape[1]] = avg
    return warped

def stitchTwo(img1, img2):

    # Find key points and SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, d1 = sift.detectAndCompute(img1, None)
    kp2, d2 = sift.detectAndCompute(img2, None)

    # Match keypoints
    good_points, good_matches = match(d1, d2)
    mat = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
    global drawCount
    cv2.imwrite('matches/matching' + str(drawCount) + '.jpg', mat)
    drawCount += 1
    print(len(good_points))
    if len(good_points) > 5:
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC)

    warped = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result = blend(warped, img1, img2)

    return result


def stitch(images, direction):
    if direction == 1:
        images = images[::-1]

    result = images[0]
    for i in range(1, len(images)):
        result = stitchTwo(result, images[i])
        rows, cols = np.where(result[:, :] != 0)
        max_col = max(cols)-5
        result = result[:, :max_col]
    return result

def main():
    print("Enter the name of txt file (with .txt): ", end='')
    txt_filename = str(input())
    print(txt_filename)
    images = load_images(txt_filename)

    print("Along which direction images will be stitched? (1 for left, 0 for right)")
    direction = int(input())

    res = stitch(images, direction)
    cv2.imwrite('result/result.jpg', res)

if __name__ == '__main__':
    main()