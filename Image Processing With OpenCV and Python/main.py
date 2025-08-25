import cv2

# img = cv2.imread("images.jpeg", 1)
img = cv2.imread(r"C:\Users\micha\OneDrive\Documents\2017codes\CV\images.jpeg")


print(img.shape)

h, w, c = img.shape
print("Dimensions of the image is:Height:", h, "pixels_Width:", w, "pixels_Number_of_Channels:", c)

# tells you the Python object type.
# It’s about the container, not the data itself
print("image type" , type(img))

# img.dtype tells you the data type of the elements inside the array.
print(img.dtype)

# prints the image as numbers (pixel intensity)
print(img)

# prints the actual image
if img is None:
    print("Error: Image not found!")  # Check if the image was loaded successfully
else:
    cv2.imshow('Parrot', img)  # Display the image in a window titled 'Parrot'
    
    k = cv2.waitKey(0)  # Wait indefinitely for a key press and store the key code in 'k'
    
    # If 'Esc' key (27) or 'q' key is pressed, close all OpenCV windows
    if k == 27 or k == ord('q'):
        cv2.destroyAllWindows()  # Safely close the image window

# Convert the image to grayscale    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  save the image
cv2.imwrite('grey_parrot.jpg', gray)
