#1 Gaussian Noise
pkg load image;
img = im2double(imread('img.png'));
gauss_img = imnoise(img, 'gaussian', 0, 0.01);
imshow(gauss_img);
title('Image with Gaussian Noise');


#Salt & Pepper Noise
sp_img = imnoise(img, 'salt & pepper', 0.05);
imshow(sp_img);
title('Image with Salt & Pepper Noise');


# Mean (Averaging) Filter
kernel = ones(3,3)/9;
mean_filtered = imfilter(gauss_img, kernel);
imshow(mean_filtered);
title('Mean Filtered Image');



#Median Filter (Best for Salt & Pepper)
if ndims(sp_img) == 3
    median_filtered = zeros(size(sp_img));
    for c = 1:3
        median_filtered(:,:,c) = medfilt2(sp_img(:,:,c));
    end
    median_filtered = im2uint8(median_filtered);
else
    median_filtered = medfilt2(sp_img);
end

imshow(median_filtered);
title('Median Filtered Image');


#Gaussian Filter
gaussian_filtered = imgaussfilt(gauss_img, 1);
imshow(gaussian_filtered);
title('Gaussian Filtered Image');


#Wiener Filter
wiener_img = wiener2(gauss_img, [5 5]);
imshow(wiener_img);
title('Wiener Filtered Image');


PSF = fspecial('motion', 15, 45);
blurred = imfilter(img, PSF, 'conv', 'circular');
restored = deconvwnr(blurred, PSF);
imshow(restored);
title('Inverse Filtered (Deblurred) Image');


#Comparing Results
mse = mean((img(:) - gaussian_filtered(:)).^2);
psnr = 10 * log10(1 / mse);
