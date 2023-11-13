import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)

    return images

# def load_images_from_folder_with_labels(folder, label):
#     images = []
#     labels = []
#     for subfolder in os.listdir(folder):
#         subfolder_path = os.path.join(folder, subfolder)
#         if os.path.isdir(subfolder_path):
#             for filename in os.listdir(subfolder_path):
#                 img_path = os.path.join(subfolder_path, filename)
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     images.append(img)
#                     labels.append(label)
#     return images, labels

def load_images_from_folder_with_labels(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels



# Function to resize images
def resize_images(images, target_size):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, target_size)
        resized_images.append(resized_image)
    return resized_images

# def extract_color_features(segmented_image):
#     # Create the CN feature object
#     cn_feature = cv2.ximgproc.createSimpleColorBalance()
#     # Compute CN features
#     cn_features = cn_feature.compute(segmented_image)

#     return cn_features

def extract_color_features(segmented_image):
    # Initialize an empty list to store the color features
    color_features = []

    # Split the segmented image into its individual channels (B, G, R)
    b, g, r = cv2.split(segmented_image)

    # Calculate the mean and standard deviation of blue and black channels
    mean_blue = np.mean(b)
    std_blue = np.std(b)

    # Black channel can be approximated as (R + G + B) / 3
    black_channel = (r + g + b) // 3
    mean_black = np.mean(black_channel)
    std_black = np.std(black_channel)

    # Append the color features to the list
    color_features.append(mean_blue)
    color_features.append(std_blue)
    color_features.append(mean_black)
    color_features.append(std_black)

    return color_features


def extract_texture_features(segmented_image):
    sift = cv2.SIFT_create()
   
    # Detect and compute SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(segmented_image, None)
   
    return keypoints, descriptors

def extract_shape_features(segmented_image):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    # Initialize lists to store shape features
    areas = []
    perimeters = []
    compactness = []
    eccentricity = []
   
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        areas.append(area)
       
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        perimeters.append(perimeter)
       
        # Calculate compactness (area / perimeter^2)
        comp = (4 * 3.1415 * area) / (perimeter ** 2)
        compactness.append(comp)
       
        # Calculate eccentricity
        moments = cv2.moments(contour)
        major_axis_length = 2 * ((moments["mu20"] + moments["mu02"] +
                                ((moments["mu20"] - moments["mu02"]) ** 2 +
                                4 * (moments["mu11"] ** 2)) ** 0.5) / 2)
        minor_axis_length = 2 * ((moments["mu20"] + moments["mu02"] -
                                ((moments["mu20"] - moments["mu02"]) ** 2 +
                                4 * (moments["mu11"] ** 2)) ** 0.5) / 2)
        eccentricity_value = (1.0 - (minor_axis_length / major_axis_length)) ** 0.5
        eccentricity.append(eccentricity_value)
   
    return areas, perimeters, compactness, eccentricity


def extract_features(segmented_image):
    # Initialize a list to store the features
    features = []

    # Split the segmented image into its individual channels (B, G, R)
    b, g, r = cv2.split(segmented_image)

    # Calculate the mean and standard deviation of blue and black channels
    mean_blue = np.mean(b)
    std_blue = np.std(b)

    # Black channel can be approximated as (R + G + B) / 3
    black_channel = (r + g + b) // 3
    mean_black = np.mean(black_channel)
    std_black = np.std(black_channel)

    # Append the color features to the list
    features.extend([mean_blue, std_blue, mean_black, std_black])

    # Create a SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(segmented_image, None)

    # You should decide how to represent these SIFT features; for now, we'll add the counts
    num_keypoints = len(keypoints)
    features.append(num_keypoints)
    num_descriptors = descriptors.shape[0] if descriptors is not None else 0
    features.append(num_descriptors)

    # Convert the segmented image to grayscale for shape analysis
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Find contours in the segmented image
    contours, _ = cv2.findContours(gray_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store shape features
    areas = []
    perimeters = []
    compactness = []
    eccentricity = []

    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        areas.append(area)

        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        perimeters.append(perimeter)

        # Avoid division by zero when calculating compactness
        if perimeter == 0:
            comp = 0
        else:
            comp = (4 * 3.1415 * area) / (perimeter ** 2)
        compactness.append(comp)

        # Calculate eccentricity
        moments = cv2.moments(contour)
        major_axis_length = 2 * ((moments["mu20"] + moments["mu02"] +
                                ((moments["mu20"] - moments["mu02"]) ** 2 +
                                4 * (moments["mu11"] ** 2)) ** 0.5) / 2)
        minor_axis_length = 2 * ((moments["mu20"] + moments["mu02"] -
                                ((moments["mu20"] - moments["mu02"]) ** 2 +
                                4 * (moments["mu11"] ** 2)) ** 0.5) / 2)
        # Avoid division by zero when calculating eccentricity_value
        if major_axis_length == 0:
            eccentricity_value = 0
        else:
            eccentricity_value = (1.0 - (minor_axis_length / major_axis_length)) ** 0.5
        eccentricity.append(eccentricity_value)

    # Append the shape features to the list
    features.extend([len(contours), np.mean(areas), np.std(areas), np.mean(perimeters), np.std(perimeters),
                     np.mean(compactness), np.std(compactness), np.mean(eccentricity), np.std(eccentricity)])

    return features

