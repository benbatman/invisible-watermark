import numpy as np
from PIL import Image
from helpers import *
import json

image = Image.open('pexels-dina-adel-19765437.jpg')
image_gray = image.convert('L')
image_data = np.array(image_gray, dtype=np.float64)

cA, cH, cV, cD = perform_dwt(image_data) 

# Generate the hash from the image data itself 
encoded_hash = generate_rc_encoded_hash_from_image(image_data)

cH_modified = embed_encoded_hash(cH, encoded_hash)
print("cH modified: ", cH_modified)
print(np.unique(cH_modified.flat))

reconstructed_image = perform_idwt(cA, cH_modified, cV, cD)
print("Dtype of reconstructed image: ", reconstructed_image.dtype)

# print(reconstructed_image)
# print("max value of image: ", max(reconstructed_image.flat))
# print("min value of image: ", min(reconstructed_image.flat))

image = Image.fromarray(np.uint8(reconstructed_image), mode='L')
image.save("dwt_image.png")


# Load the potentially modified image
image_modified = Image.open('dwt_image.png')
image_modified_gray = image_modified.convert('L')
image_modified_data = np.array(image_modified_gray, dtype=np.float64)

# # Check to make sure image values are the same
# print(image_modified_data)
# print("max value of image: ", max(image_modified_data.flat))
# print("min value of image: ", min(image_modified_data.flat))


#Perform DWT on the modified image
cA_modified, cH_modified_from_saved, cV_modified, cD_modified = perform_dwt(image_modified_data)
print("cH modified: ", cH_modified_from_saved)

# Check cH_modified to make sure it is the same as cH
print("Max of cH modified: ", max(cH_modified_from_saved.flat))
print("Min of cH modified: ", min(cH_modified_from_saved.flat))
# Extract the hash from the modified cH
extracted_hash = extract_hash_dwt_rc(cH_modified_from_saved)
print("Extracted hash: ", extracted_hash)