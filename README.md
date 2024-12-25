import cv2
import numpy as np
import os

# Load the uploaded image
input_image_path = "/mnt/data/image.png"
output_dir = "/mnt/data/output/"
os.makedirs(output_dir, exist_ok=True)

# Read the image
image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError("Image could not be loaded!")

# 1. Shift image
shift_matrix = np.float32([[1, 0, 10], [0, 1, 20]])
shifted_image = cv2.warpAffine(image, shift_matrix, (image.shape[1], image.shape[0]))
cv2.imwrite(output_dir + "shifted_image.jpg", shifted_image)

# 2. Inversion
inverted_image = cv2.bitwise_not(image)
cv2.imwrite(output_dir + "inverted_image.jpg", inverted_image)

# 3. Gaussian blur
gaussian_blur = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imwrite(output_dir + "gaussian_blur.jpg", gaussian_blur)

# 4. Motion blur (diagonal)
def motion_blur_kernel(size):
    kernel = np.zeros((size, size))
    np.fill_diagonal(kernel, 1)
    return kernel / size

motion_kernel = motion_blur_kernel(7)
motion_blurred = cv2.filter2D(image, -1, motion_kernel)
cv2.imwrite(output_dir + "motion_blur.jpg", motion_blurred)

# 5. Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
cv2.imwrite(output_dir + "sharpened_image.jpg", sharpened_image)

# 6. Sobel filter (X direction)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
cv2.imwrite(output_dir + "sobel_filter.jpg", sobel_x)

# 7. Edge detection (Canny)
edges = cv2.Canny(image, 100, 200)
cv2.imwrite(output_dir + "edges.jpg", edges)

# 8. Custom filter (Emboss)
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
embossed_image = cv2.filter2D(image, -1, emboss_kernel)
cv2.imwrite(output_dir + "embossed_image.jpg", embossed_image)

output_dir
