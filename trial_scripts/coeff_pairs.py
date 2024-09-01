import numpy as np
from PIL import Image
from helpers import *


image = Image.open('pexels-dina-adel-19765437.jpg')
image_gray = image.convert('L')
image_data = np.array(image_gray, dtype=np.float64)

cA, cH, cV, cD = perform_dwt(image_data) 

# Generate the hash from the image data itself 
encoded_hash = generate_rc_encoded_hash_from_image(image_data)

coeff_pairs = create_coeff_pairs(cH, encoded_hash)
print("length of coeff pairs: ", len(coeff_pairs))
cH_modified = embed_with_pairs(cH, encoded_hash, coeff_pairs)

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


#Perform DWT on the modified image
cA_modified, cH_modified_from_saved, cV_modified, cD_modified = perform_dwt(image_modified_data)
# print("cH modified: ", cH_modified_from_saved)
coeff_pairs_modified = create_coeff_pairs(cH_modified_from_saved, encoded_hash)
print("length of coeff pairs: ", len(coeff_pairs))


# Extract the hash from the modified cH
extracted_hash = extract_with_pairs(cH_modified_from_saved, coeff_pairs_modified)
print("Extracted hash: ", extracted_hash)