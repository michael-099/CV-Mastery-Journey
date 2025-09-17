img = imread('img.png');
imshow(img);
title('Original Image');


%Binary Segmentation using a Fixed Threshold
img = imread('img.png');
bw_img = img > 100;
imshow(uint8(bw_img) * 255);
title('Binary Image (Threshold = 100)');

%Otsu’s Method for Adaptive Thresholding
if ndims(img) == 3
    img = rgb2gray(img);
end

level = graythresh(img);
bw_otsu = img > level * 255;
imshow(bw_otsu);
title("Otsu's Thresholding");

%Edge-Based Segmentation
edge_sobel = edge(img, 'sobel');
imshow(edge_sobel);
title('Sobel Edge Detection');

%Canny Edge Detection
edge_canny = edge(img, 'canny');
imshow(edge_canny);
title('Canny Edge Detection');

pkg load image;
I = im2double(imread('img.png'));
BW = im2bw(I, 0.4);
label = bwlabel(BW);
imshow(label, []);
title('Region Labels after Thresholding');


pkg load signal; % Required for DWT in Octave
img = imread('img.png');
if ndims(img) == 3
    img = rgb2gray(img);
end
img = im2double(img);

wavelets = {'db2', 'sym4'}; % Different wavelet bases

for i = 1:length(wavelets)
    wname = wavelets{i};
    [cA, cH, cV, cD] = dwt2(img, wname);

    figure;
    subplot(2,2,1); imshow(cA, []); title(['Approximation (' wname ')']);
    subplot(2,2,2); imshow(cH, []); title('Horizontal Detail');
    subplot(2,2,3); imshow(cV, []); title('Vertical Detail');
    subplot(2,2,4); imshow(cD, []); title('Diagonal Detail');
end

