# Identify characters from each image and map "character: filename" to a dictionary
"""Uses keras-ocr to identify characters, then creates a dictionary where each entry is
   a character and its respective image (file name)."""

import os
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline() # Load OCR model
path = 'img/char'                        # Path to char images
files_and_chars = {}                     # Dict stores "character: filename"

# Iterate through each char image in path
for filename in os.listdir(path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = keras_ocr.tools.read(os.path.join(path, filename)) # Load image
        results = pipeline.recognize([image])                      # Recognize char in image

        # Extract recognized character and add to dict
        try:
            character = results[0][0][0]
            if character in files_and_chars:
                print("Already in dict")
                num = 0
                while character in files_and_chars:
                    character += str(num)
                    num += 1
                files_and_chars[character] = filename
            else:
                files_and_chars[character] = filename
        except:
            continue
            
print(files_and_chars)