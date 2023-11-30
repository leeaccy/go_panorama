# This is the code for project Go Panorama in EECS 504 23 Fall at UMich

import cv2
import os
from tools import sift_blend

def on(source_folder):

    images = []
    files = os.listdir(source_folder)
    file_no = 0
    for i in range(0, len(files)):
        position = source_folder + '/' + str(file_no) + '.jpg'
        if os.path.exists(position):
            images.append(cv2.imread(position))

    # initial the function
    image1 = cv2.imread(source_folder + '/0' + '.jpg')  # right
    image2 = cv2.imread(source_folder + '/1' + '.jpg')  # left
    result = sift_blend(image1, image2)
    print('image 0')
    print('image 1')

    # iteration
    for i in range(2, len(images)):
        image1 = result
        image2 = cv2.imread(source_folder + '/' + str(i) + '.jpg')
        result = sift_blend(image1, image2)
        print('image', i)

    print('Done.')

    cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stitched Image', 1920, 1080)
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('Go Panorama!')

    # stitch images
    on('./Dataset/Dataset5')  # Change the dataset you want to use here

