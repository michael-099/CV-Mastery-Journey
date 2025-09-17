## Canny Edge Detector
pkg load image;
img = im2double(imread('img.png'));

if ndims(img) == 3
    img = rgb2gray(img);   % convert RGB → grayscale
end

edges = edge(img, 'canny');
imshow(edges);
title('Canny Edge Detection');


## Harris Corner Detector
% Parameters
sigma = 2;    % Gaussian smoothing
k = 0.04;     % Harris constant
threshold = 1e-5;

% Compute gradients
[Gx, Gy] = imgradientxy(img);

% Structure tensor components
Ix2 = imgaussfilt(Gx.^2, sigma);
Iy2 = imgaussfilt(Gy.^2, sigma);
Ixy = imgaussfilt(Gx.*Gy, sigma);

% Harris response
R = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;

% Thresholding
corner_map = R > threshold * max(R(:));

% Display
imshow(img); hold on;
[y, x] = find(corner_map);
plot(x, y, 'r.');
title('Harris Corners');

## Marking Detected Corners
pkg load image;

img = im2double(imread('img.png'));
if ndims(img) == 3
    img = rgb2gray(img);
end

% Parameters
sigma = 2;    % Gaussian smoothing
k = 0.04;     % Harris constant
threshold = 0.01; % relative threshold

% Compute gradients
[Gx, Gy] = imgradientxy(img);

% Structure tensor components (smoothed)
Ix2 = imgaussfilt(Gx.^2, sigma);
Iy2 = imgaussfilt(Gy.^2, sigma);
Ixy = imgaussfilt(Gx.*Gy, sigma);

% Harris response
R = (Ix2 .* Iy2 - Ixy.^2) - k * (Ix2 + Iy2).^2;

% Threshold Harris response
corner_thresh = R > threshold * max(R(:));

% Get row/col coordinates
[r, c] = find(corner_thresh);

% Show detected corners
imshow(img); hold on;
plot(c, r, 'r*');
title('Detected Corners');

## Blob Detection using LoG (Laplacian of Gaussian)
log_filter = fspecial('log', [5 5], 0.5);
blob_img = imfilter(img, log_filter, 'replicate');
imshow(blob_img, []);
title('Blob Detection via LoG');


# Feature Descriptor (Simplified - Intensity Patch)
patch1 = img(30:39, 40:49); % Patch from reference image
imshow(patch1);
title('Template Patch');

#Template Matching (Correlation-Based)
corr_result = normxcorr2(patch1, img);
[max_corr, idx] = max(corr_result(:));
[y, x] = ind2sub(size(corr_result), idx);
imshow(img);
hold on;
rectangle('Position', [x-9, y-9, 10, 10], 'EdgeColor', 'g');
title('Matched Template Location');

