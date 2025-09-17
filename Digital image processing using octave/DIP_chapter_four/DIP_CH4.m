img = imread('img.png');
imshow(img);
title('Original Image');

%Reading and Binarizing an Image
gray = rgb2gray(img);
binary_img = im2bw(gray, 0.5);
imshow(binary_img);
title('Binary Input Image');


% Creating a Structuring Element

se = strel('square', 3); % 3x3 square structuring element


#Basic Morphological Operations
#Erosion
eroded_img = imerode(binary_img, se);
imshow(eroded_img);
title('Eroded Image');

# Dilation
dilated_img = imdilate(binary_img, se);
imshow(dilated_img);
title('Dilated Image');

#Closing
closed_img = imclose(binary_img, se);
imshow(closed_img);
title('Closed Image')




