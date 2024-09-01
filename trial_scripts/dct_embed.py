import numpy as np
import cv2 
from PIL import Image
from helpers import extract_token_from_image
from cryptography.fernet import Fernet

# Some studies that look at using DCT and other methods to embed a watermark
# https://ieeexplore.ieee.org/document/9847745
# https://ieeexplore.ieee.org/document/9664853


key = Fernet.generate_key()
f = Fernet(key)
# Create the token
token = f.encrypt(b"some manifest?") # I think this would be the cryptographic manifest hash 
print(f"Token: {token} | Token Type: {type(token)}")

def process_image_and_apply_dct(ycrcb_image):
    print("YCrCb image shape:", ycrcb_image.shape)
    # Get the y channel, or luminance channel of the image
    y_channel = ycrcb_image[:, :, 0].astype(float)
    print("Y channel:", y_channel)
    # Get height and width of the image
    h, w = y_channel.shape 
    print("Height:", h, "\nWidth:", w)
    dct_channel = np.zeros_like(y_channel, dtype=float) # Will be modifying the luminance channel
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = y_channel[i:i+8, j:j+8]
            dct_block = cv2.dct(block.astype(float))
            dct_channel[i:i+8, j:j+8] = dct_block
    
    return dct_channel

image_path = "pexels-dina-adel-19765437.jpg"
image = cv2.imread(image_path)
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
dct_y_channel = process_image_and_apply_dct(ycrcb_image) 


# Convert the token to a binary string 
token_bits = ''.join(format(byte, '08b') for byte in token)
token_bits_length = len(token_bits)
print(token_bits)
print("Token bits length:", token_bits_length)

# Embed the token bits into the DCT Coefiicients 
for i, bit in enumerate(token_bits):
    row, col = divmod(i, dct_y_channel.shape[1]) # Get the row and column of the coefficient to modify
    coeff = dct_y_channel[row, col]
    # Embed the bit, Changing the whole coefficient by rounding
    if bit == '1':
        dct_y_channel[row, col] = np.ceil(coeff)
    else:
        dct_y_channel[row, col] = np.floor(coeff)


### Apply IDCT to the modified DCT coefficients
def apply_idct(dct_channel):
    h, w = dct_channel.shape
    idct_channel = np.zeros_like(dct_channel)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            dct_block = dct_channel[i:i+8, j:j+8]
            idct_block = cv2.idct(dct_block.astype(float)) # inverse dct
            idct_channel[i:i+8, j:j+8] = idct_block
    return idct_channel 

print("Applying IDCT...")
modified_y_channel = apply_idct(dct_y_channel)
modified_y_channel = np.clip(modified_y_channel, 0, 255).astype(np.uint8)
print("Modified Y channel:", modified_y_channel)

# Reconstruct the image with the modified Y channel
ycrcb_image[:, :, 0] = modified_y_channel # Set the luminance channel to the modified Y channel
print("Reconstructed image shape:", image.shape)
modified_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
modified_image = np.clip(modified_image, 0, 255).astype('uint8')
print("Successfully modified the image")


#Get the token back from the image 
retrieved_token_bytes = extract_token_from_image(modified_image, token_bits_length)
print("Retrieved token bits:", retrieved_token_bytes)
print("Retrieved token bits length:", len(retrieved_token_bytes))
# Convert the retrieved token bytes to a binary string
# binary_token = ''.join(format(byte, '08b') for byte in retrieved_token_bytes)
# print("Binary token:", binary_token)
# Compare binary strings to ensure the token is correct
if retrieved_token_bytes == token_bits:
    print("Token is correct!")
else:
    print("Tokens do not match")

# Conver the binary string to a byte string
# num_bytes = len(binary_token) // 8 
# retrieved_token = int(binary_token, 2).to_bytes(num_bytes, byteorder='big')
# print("Retrieved token:", retrieved_token)


# byte_chunks = [retrieved_token_bytes[i:i+8] for i in range(0, len(retrieved_token_bytes), 8)]
# retrieved_token = bytes(int(byte, 2) for byte in byte_chunks)
# decrypted_token = f.decrypt(retrieved_token)
# print("Decrypted token:", decrypted_token)
# # Ensure the token is correct
# if decrypted_token == token:
#     print("Token is correct!")
# else:
#     print("Token is incorrect!")

# Save the watermarked image
print("Saving the watermarked image...")
cv2.imwrite("watermarked_image.png", modified_image)