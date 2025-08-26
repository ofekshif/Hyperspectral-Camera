import cv2
import numpy as np
import os
image1_path = r'C:\Users\ofeks\PycharmProjects\picturesforwebcamerashift\image1.jpg'
print(os.path.isfile(image1_path))
image2_path = r'C:\Users\ofeks\PycharmProjects\picturesforwebcamerashift\image2.jpg'
def find_shift(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread('image1.jpg',cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('image2.jpg',cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        raise ValueError("Error loading images. Check file paths.")

    # Step 1: Detect keypoints and descriptors using ORB
    img1 = np.float32(image1)
    img2 = np.float32(image2)

    # Calculate the shift using phase correlation
    shift, _ = cv2.phaseCorrelate(img1, img2)

    return shift


# Example usage
image1 = "path_to_first_image.jpg"
image2 = "path_to_second_image.jpg"
shift = find_shift(image1, image2)

print("Shift (in pixels):")
print(f"Horizontal: {shift[0]}")
print(f"Vertical: {shift[1]}")

# Example usage
if __name__ == "__main__":
    image1_path = "image1.jpg"  # Path to the first image
    image2_path = "image2.jpg"  # Path to the second image


