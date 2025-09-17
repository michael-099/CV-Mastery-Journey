#Constructing a Gaussian Pyramid
pkg load image;
img = im2double(imread('img.png'));
levels = 4;
g_pyramid = cell(1, levels);
g_pyramid{1} = img;
for i = 2:levels
g_pyramid{i} = impyramid(g_pyramid{i-1}, 'reduce');
end
#figure;
for i = 1:levels
subplot(1,levels,i);
imshow(g_pyramid{i});
title(['Level ', num2str(i)]);
end



#Constructing a Laplacian Pyramid
l_pyramid = cell(1, levels-1);
for i = 1:levels-1
upsampled = impyramid(g_pyramid{i+1}, 'expand');
upsampled = imresize(upsampled, size(g_pyramid{i})(1:2));

l_pyramid{i} = g_pyramid{i} - upsampled;
end
#figure;
for i = 1:levels-1
subplot(1,levels-1,i);
imshow(l_pyramid{i}, []);
title(['Laplacian Level ', num2str(i)]);
end


#Image Reconstruction from Laplacian Pyramid
reconstructed = g_pyramid{levels};
for i = levels-1:-1:1
upsampled = impyramid(reconstructed, 'expand');
upsampled = imresize(upsampled, size(l_pyramid{i})(1:2));
reconstructed = l_pyramid{i} + upsampled;
end
imshow(reconstructed);
title('Reconstructed Image from Laplacian Pyramid');

#Decomposition using DWT
pkg load wavelet;
[LL, LH, HL, HH] = dwt2(img, 'haar');
figure;
subplot(2,2,1), imshow(LL, []), title('Approximation');
subplot(2,2,2), imshow(LH, []), title('Horizontal');
subplot(2,2,3), imshow(HL, []), title('Vertical');
subplot(2,2,4), imshow(HH, []), title('Diagonal');
