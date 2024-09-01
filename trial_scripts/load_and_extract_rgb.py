from PIL import Image
from helpers import *

####### Load the image back in and extract the hash #######

hash_length = 416

image = Image.open("dwt_image_rgb.jpeg") 
image_data = np.array(image, dtype=np.float64)

ycrcb_image = cv2.cvtColor(np.clip(image_data, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
y_channel = ycrcb_image[:, :, 0].astype(np.float64)

# Get each color channel 
R, G, B = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]

# Perform DWT on each color channel
cA_R, cH_R, cV_R, cD_R = perform_dwt(R)
cA_G, cH_G, cV_G, cD_G = perform_dwt(G)
cA_B, cH_B, cV_B, cD_B = perform_dwt(B)
cA_Y, cH_Y, cV_Y, cD_Y = perform_dwt(y_channel)

# Generate the coefficient pairs
coeff_pairs_R = create_coeff_pairs(cH_R, hash_length)
coeff_pairs_G = create_coeff_pairs(cH_G, hash_length)
coeff_pairs_B = create_coeff_pairs(cH_B, hash_length)
coeff_pairs_Y = create_coeff_pairs(cH_Y, hash_length)

# Extract the hash from the cH channels for each color channel
extracted_hash_R = extract_with_pairs(cH_R, coeff_pairs_R)
extracted_hash_G = extract_with_pairs(cH_G, coeff_pairs_G)
extracted_hash_B = extract_with_pairs(cH_B, coeff_pairs_B)
extracted_hash_Y = extract_with_pairs(cH_Y, coeff_pairs_Y)

# print("Original hash value: ", encoded_hash)
print("\n\nExtracted hash value R: ", extracted_hash_R)
print("\n\nExtracted hash value G: ", extracted_hash_G)
print("\n\nExtracted hash value B: ", extracted_hash_B)
print("\n\nExtracted hash value Y: ", extracted_hash_Y)