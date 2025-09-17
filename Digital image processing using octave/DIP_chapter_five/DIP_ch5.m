pkg load image;
img = imread('img.png');
img = im2double(img);
imshow(img);
title('Original Image');

#raging (Box) Filter
#Applying a 3x3 Averaging Filter
kernel = ones(3,3)/9;
avg_filtered = imfilter(img, kernel);
imshow(avg_filtered);
title('Averaging Filter (3x3)');

#Applying Gaussian Filter
#Gaussian Filtering
gauss_filtered = imgaussfilt(img, 1);
imshow(gauss_filtered);
title('Gaussian Filter (Sigma = 1)');



#Median Filtering
pkg load image
img = imread('img.png');

% Convert to grayscale if RGB
if ndims(img) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Apply median filter
median_filtered = medfilt2(img_gray);
imshow(median_filtered);
title('Median Filtered Image');


laplacian_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
laplacian_img = imfilter(img, laplacian_kernel);
sharpened_img = img + laplacian_img;
imshow(sharpened_img);
title('Sharpened Image (Laplacian)');



%erforming 2D Correlation
% Define your kernel (example: 3x3 averaging filter)
kernel = ones(3,3) / 9;
% Apply convolution
corr_output = conv2(double(img_gray), kernel, 'same');
% Display result
imshow(corr_output, []);
title('Correlation Output');


sobel_x = [-1 0 1; -2 0 2; -1 0 1];
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
gx = imfilter(img, sobel_x);
gy = imfilter(img, sobel_y);
gradient_magnitude = sqrt(gx.^2 + gy.^2);
imshow(gradient_magnitude, []);
title('Edge Detection using Sobel');
