img = imread('img.png');
imshow(img);
title('Original Image');

#{
sampled_img = img(1:2:end, 1:2:end);
imshow(sampled_img);
title('Sampled Image - Factor 2');

sampled_img_4 = img(1:4:end, 1:4:end);
imshow(sampled_img_4);
title('Sampled Image - Factor 4');

quant_img = uint8(floor(double(img)/64) * 64);
imshow(quant_img);
title('Quantized Image - 4 Gray Levels');


binary_img = img > 128;
imshow(uint8(binary_img) * 255);
title('Binary Image (Threshold at 128)');


binary_img = img > 12;
imshow(binary_img);
title('Binary Image (Threshold at 128)');

cascaded_img = img(1:2:end, 1:2:end);
cascaded_img = uint8(floor(double(cascaded_img)/64) * 64);
figure;
imshow(cascaded_img);
title('Sampled and Quantized Image');

%% 1. Downsampling with different factors



%% 2. Quantization with different gray levels


%% 3. Binary image (threshold at 128)
binary_img = img > 128;
figure;
imshow(binary_img);
title('Binary Image (Threshold at 128)');

%% 4. Cascaded sampling and quantization (factor 2 + 4 gray levels)
cascaded_img = img(1:2:end, 1:2:end);
cascaded_img = uint8(floor(double(cascaded_img)/64) * 64);
figure;
imshow(cascaded_img);
title('Sampled (Factor 2) + Quantized (4 Levels)');
#}

gray_levels = [4, 8, 16, 32];
figure;
for i = 1:length(gray_levels)
    q = gray_levels(i);
    step = 256 / q;
    quant_img = uint8(floor(double(img)/step) * step);
    subplot(2, 2, i);
    imshow(quant_img);
    title(sprintf('%d Gray Levels', q));
end
factors = [3, 4, 5];
figure;
for i = 1:length(factors)
    sampled_img = img(1:factors(i):end, 1:factors(i):end);
    subplot(1, length(factors), i);
    imshow(sampled_img);
    title(sprintf('Factor %d', factors(i)));
end

