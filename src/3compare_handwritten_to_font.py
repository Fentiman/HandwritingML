# Compare handwritten characters from dictionary to font characters

import os
import imagehash
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt

print(files_and_chars)

# Iterate through each char image in path
for key, val in files_and_chars.items():
    written_path = 'img/char'
    written_path += ("/" + val)
    
    font_path = 'fonts/hynings'           # Path to font images
    font_path += ("/" + key[0] + ".png") # Append filename to path
    
    written_char = io.imread(written_path, as_gray=True) # Handwritten character
    font_char    = io.imread(font_path, as_gray=True)    # Typed character
    
    # Convert NumPy arrays to PIL Image objects
    written_char = Image.fromarray((written_char * 255).astype('uint8'))
    font_char = Image.fromarray((font_char * 255).astype('uint8'))
    
    hash1 = imagehash.average_hash(written_char)
    hash2 = imagehash.average_hash(font_char)
    
    distance = hash1 - hash2
    
    print(f"The hamming distance between the two images is {distance}")
    
    # Plot the images and the hamming distance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(written_char, cmap='gray')
    ax1.set_title('Written character')
    ax2.imshow(font_char, cmap='gray')
    ax2.set_title('Font character')
    plt.suptitle(f"Hamming distance: {distance}", fontsize=16)
    plt.show()