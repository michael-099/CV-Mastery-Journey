import matplotlib.pyplot as plt
import numpy as np
import cv2


def extract_bit_plane(cd):
    c1 = np.mod(cd, 2)
    c2 = np.mod(np.floor(cd/2), 2)
    c3 = np.mod(np.floor(cd/4), 2)
    c4 = np.mod(np.floor(cd/8), 2)
    c5 = np.mod(np.floor(cd/16), 2)
    c6 = np.mod(np.floor(cd/32), 2)
    c7 = np.mod(np.floor(cd/64), 2)
    c8 = np.mod(np.floor(cd/128), 2)
    
    # combining image again to form equivalent to original grayscale image 
    cc = 2 * (2 * (2 * c8 + c7) + c6) # reconstructing image  with 3 most significant bit planes
    to_plot = [cd, c1, c2, c3, c4, c5, c6, c7, c8, cc]
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(10, 8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, i in zip(axes.flat, to_plot):
        ax.imshow(i, cmap='gray')
    plt.tight_layout()
    plt.show()
    return cc
gray = cv2.imread(r"C:\Users\micha\OneDrive\Documents\2017codes\CV\grey_parrot.jpg")
reconstructed_image = extract_bit_plane(gray)

# Create a blank 256x256 image (all zeros = black)
con_img = np.zeros([256, 256])

# Outer rectangle border (thickness: 32 pixels)
con_img[0:32, :] = 40       # Top border
con_img[:, :32] = 40        # Left border
con_img[:, 224:256] = 40    # Right border
con_img[224:, :] = 40       # Bottom border

# Second rectangle border
con_img[32:64, 32:224] = 80     # Top
con_img[64:224, 32:64] = 80     # Left
con_img[64:224, 192:224] = 80   # Right
con_img[192:224, 32:224] = 80   # Bottom

# Third rectangle border
con_img[64:96, 64:192] = 160    # Top
con_img[96:192, 64:96] = 160    # Left
con_img[96:192, 160:192] = 160  # Right
con_img[160:192, 64:192] = 160  # Bottom

# Innermost square
con_img[96:160, 96:160] = 220

# Display the image using matplotlib
plt.imshow(con_img, cmap='gray')  # Use grayscale colormap
plt.title("Concentric Rectangles")
plt.axis('off')  # Hide axes
plt.show()
