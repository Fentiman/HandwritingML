import cv2
from imutils import contours

# Load image
image = cv2.imread('one_line.png')
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Thresholded using Otsu's 
# (Separates background from foreground i.e. create a binary image)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours, sort from left-to-right, then crop
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

char_number = 0 # Assign number to each character
for c in cnts: # For each contour (each character)
    area = cv2.contourArea(c) # Get area of contour
    if area > 100: # Area must be large enough to filter out noise
        x,y,w,h = cv2.boundingRect(c) # Get coordinates of character
        char = 255 - image[y:y+h, x:x+w] # Invert BW
        cv2.imwrite('png_example_characters/char_{}.png'.format(char_number), char) # Save character as a PNG
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2) # Draw rectangle around character
        char_number += 1

cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()