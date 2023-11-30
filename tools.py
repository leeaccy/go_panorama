# This is the code for project Go Panorama in EECS 504 23 Fall at UMich

import cv2
import numpy as np

def sift_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Keep good matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Find homography using RANSAC
    H, status = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    # Warp the images
    result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], int(1.1 * max(image1.shape[0], image2.shape[0]))))

    result[0:image2.shape[0], 0:image2.shape[1]] = image2

    return result

def sift_blend(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Keep good matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Find homography using RANSAC
    H, status = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    # Warp the first image
    warped_image1 = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], int(1.1 * max(image1.shape[0], image2.shape[0]))))

    # Ensure both images have the same height for blending
    h1, w1 = warped_image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # If the heights are different, pad the smaller image vertically
    if h1 != h2:
        if h1 > h2:
            # Pad image2
            padding = h1 - h2
            image2 = cv2.copyMakeBorder(image2, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            # Pad warped_image1
            padding = h2 - h1
            warped_image1 = cv2.copyMakeBorder(warped_image1, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Create an empty canvas for the result
    result = np.zeros_like(warped_image1)

    # Calculate the width for blending
    blend_width = int(0.2 * w2)  # Adjust as needed

    # Set the region for blending
    start = w2 - blend_width
    end = w2

    # Blend the images
    for i in range(start, end):
        alpha = (i - start) / (end - start)
        result[:, i] = cv2.addWeighted(warped_image1[:, i], alpha, image2[:, i] if i < w2 else 0, 1 - alpha, 0)

    # Copy non-overlapping parts
    result[0:h2, 0:start] = image2[0:h2, 0:start]
    result[0:h1, end:] = warped_image1[0:h1, end:]

    return result

def stitch_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters and matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Keep good matches using Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Find homography using RANSAC
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    # Warp the first image
    warped_image1 = cv2.warpPerspective(image1, H, (image2.shape[1] + image1.shape[1], image2.shape[0]))

    # Adjust heights of images if different
    h1, w1 = warped_image1.shape[:2]
    h2, w2 = image2.shape[:2]

    if h1 != h2:
        if h1 > h2:
            image2 = cv2.copyMakeBorder(image2, 0, h1 - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            warped_image1 = cv2.copyMakeBorder(warped_image1, 0, h2 - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Blend images
    result = np.zeros(warped_image1.shape, dtype=warped_image1.dtype)
    blend_width = int(0.2 * w2)  # Adjust as needed
    start = w2 - blend_width
    end = w2

    for i in range(start, end):
        alpha = (i - start) / blend_width
        result[:, i] = cv2.addWeighted(warped_image1[:, i], alpha, image2[:, i], 1 - alpha, 0)

    # Copy non-overlapping parts
    result[:, :start] = image2[:, :start]
    result[:, end:] = warped_image1[:, end:]

    return result

def on_images(images):

    # initial the function
    image1 = images[0]  # right
    image2 = images[1]  # left
    result = sift_images(image1, image2)

    # iteration
    for i in range(2, len(images)):
        print('image' ,i)
        image1 = result
        image2 = images[i]
        result = sift_images(image1, image2)

    cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stitched Image', 1920, 1080)
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






