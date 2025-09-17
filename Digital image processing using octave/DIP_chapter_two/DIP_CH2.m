color_img = imread('img.png');
imshow(color_img);
title('Original Color Image');

R = color_img(:,:,1);
G = color_img(:,:,2);
B = color_img(:,:,3);
subplot(1,3,1); imshow(R); title('Red Channel');
subplot(1,3,2); imshow(G); title('Green Channel');
subplot(1,3,3); imshow(B); title('Blue Channel');

bright_img = color_img + 50;
imshow(bright_img);
title('Brightness Increased');

dark_img = color_img - 50;
imshow(dark_img);
title('Brightness Decreased');

negative_img = 255 - color_img;
imshow(negative_img);
title('Negative Image');

double_img = im2double(color_img);
contrast_img = imadjust(double_img, stretchlim(double_img), [0 1]);
imshow(contrast_img);
title('Contrast Stretched Image');

gray_img = 0.2989 * double(color_img(:,:,1)) + 0.5870 * double(color_img(:,:,2)) + 0.1140 * double(color_img(:,:,3));
gray_img = uint8(gray_img);
imshow(gray_img);
title('Grayscale Image (Luminosity Method)');

R_eq = histeq(color_img(:,:,1));
G_eq = histeq(color_img(:,:,2));
B_eq = histeq(color_img(:,:,3));

hist_eq_img = cat(3, R_eq, G_eq, B_eq);
imshow(hist_eq_img);
title('Histogram Equalized Color Image');


function dynamic_brightness()
    fig = figure('Name','Dynamic Brightness Adjustment');
    img = imread('img.png');
    h_img = imshow(img);

    % Slider for brightness [-100,100]
    uicontrol('Style','slider', ...
              'Min',-100,'Max',100,'Value',0, ...
              'Position',[150 20 300 20], ...
              'Callback', @(src,~) adjust_brightness(src.Value, img, h_img));
end

function adjust_brightness(value, img, h_img)
    bright_img = uint8(double(img) + value);
    set(h_img, 'CData', bright_img);
end

#{













