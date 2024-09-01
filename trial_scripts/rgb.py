import numpy as np
from PIL import Image
from helpers import *


# Load RGB image 
image = Image.open("pexels-dina-adel-19765437.jpg")
image_data = np.array(image, dtype=np.float64) 

# Generate the hash from the image data (this could be any hash)
encoded_hash, bit_hash, hash_length = generate_rc_encoded_hash_from_image(image_data)

# Seprate the RGB channels 
R, G, B = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]

# Apply DWT to each channel 
cA_R, cH_R, cV_R, cD_R = perform_dwt(R)
cA_G, cH_G, cV_G, cD_G = perform_dwt(G)
cA_B, cH_B, cV_B, cD_B = perform_dwt(B)
print("shape of cH: ", cH_R.shape)


# Generate the coefficient pairs for color channels and Y channel
coeff_pairs_R = create_coeff_pairs(cH_R, hash_length)
coeff_pairs_G = create_coeff_pairs(cH_G, hash_length)
coeff_pairs_B = create_coeff_pairs(cH_B, hash_length)

# Embed hash into the cH channels for each color channel and Y channel
cH_R_modified = embed_with_pairs(cH_R, bit_hash, coeff_pairs_R)
cH_G_modified = embed_with_pairs(cH_G, bit_hash, coeff_pairs_G)
cH_B_modified = embed_with_pairs(cH_B, bit_hash, coeff_pairs_B)

# Reconstruct each color channel 
R_reconstructed = perform_idwt(cA_R, cH_R, cV_R, cD_R)
G_reconstructed = perform_idwt(cA_G, cH_G, cV_G, cD_G)
B_reconstructed = perform_idwt(cA_B, cH_B, cV_B, cD_B)

# Combine the RGB Channels 
reconstructed_image = np.stack((R_reconstructed, G_reconstructed, B_reconstructed), axis=-1)

# Ensure pixel values are within the range of 0-255 and uint8
reconstructed_image = np.clip(reconstructed_image, 0, 255)
reconstructed_image_uint8 = np.uint8(reconstructed_image)

#### Perform Y Channel modifications #### 

# Use the RGB reconstructed image as the base to alter the Y channel
# Convert the image into the YCrCb color space
ycrcb_image = cv2.cvtColor(reconstructed_image_uint8, cv2.COLOR_BGR2YCrCb)
# Get the Y channel
y_channel = ycrcb_image[:, :, 0].astype(np.float64)
print("shape of y channel: ", y_channel.shape)

cA_Y, cH_Y, cV_Y, cD_Y = perform_dwt(y_channel)
coeff_pairs_Y = create_coeff_pairs(cH_Y, hash_length)
cH_Y_modified = embed_with_pairs(cH_Y, bit_hash, coeff_pairs_R)
y_reconstructed = perform_idwt(cA_Y, cH_Y_modified, cV_Y, cD_Y)
ycrcb_modified_image = np.stack((y_reconstructed, ycrcb_image[:, :, 1], ycrcb_image[:, :, 2]), axis=-1)
final_rgb_image = cv2.cvtColor(ycrcb_modified_image.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

### End Y Channel modifications ####


# Create a PIL image from the numpy array 
reconstructed_image = Image.fromarray(final_rgb_image, mode='RGB')

# Save the reconstructed image as PNG
reconstructed_image.save("dwt_image_rgb.png")
# Save the modified cH channel as JPEG 
reconstructed_image.save("dwt_image_rgb.jpeg")
