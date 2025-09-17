#Simulating a Transformed Image
pkg load image;
fixed = im2double(imread('img.png'));
% Simulate moving image (rotated + translated)
moving = imrotate(fixed, 10, 'bilinear', 'crop');
moving = circshift(moving, [10, 15]);
figure;
subplot(1,2,1); imshow(fixed); title('Fixed Image');
subplot(1,2,2); imshow(moving); title('Moving Image');


# Simulating a Transformed Image
[mp, fp] = cpselect(moving, fixed, 'Wait', true);

#Estimating the Geometric Transformation
% Example: manually specify corresponding points
mp = [15 30;  60 80;  120 150];
fp = [10 25;  55 75;  115 145];

tform = cp2tform(mp, fp, 'affine');
registered = imtransform(moving, tform, 'XData', [1 size(fixed,2)], 'YData', [1 size(fixed,1)]);
imshowpair(fixed, registered, 'montage');

imshowpair(fixed, registered, 'montage');


corners1 = detectHarrisFeatures(fixed);
corners2 = detectHarrisFeatures(moving);

error_map = abs(fixed - registered);
imshow(error_map, []);
title('Registration Error Map');
