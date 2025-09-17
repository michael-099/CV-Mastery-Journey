
#Load Required Package and Imagepkg load image;
img = im2double(imread('img.png'));
imshow(img);
title('Original Image');

#Applying 2D Fourier Transform
F = fft2(img);
F_shifted = fftshift(F);
F_magnitude = log(1 + abs(F_shifted));
imshow(F_magnitude, []);
title('Magnitude Spectrum of DFT');

#Inverse Fourier Transform
reconstructed = real(ifft2(F));
imshow(reconstructed);
title('Reconstructed Image from DFT');

pkg load signal

img = rgb2gray(img);
img = im2double(img);

function out = dct2_custom(in)
    out = dct(dct(in)')';
end

function out = idct2_custom(in)
    out = idct(idct(in)')';
end

# Applying DCT
dct_img = dct2_custom(img);
imshow(log(abs(dct_img) + 1), []);
title('DCT Coefficients');

# Inverse DCT
reconstructed_dct = idct2_custom(dct_img);
imshow(reconstructed_dct, []);
title('Reconstructed Image from DCT');


pkg install -forge wavelet;
pkg load wavelet;
[LL, LH, HL, HH] = dwt2(img, 'haar');
subplot(2,2,1), imshow(LL, []), title('Approximation (LL)');
subplot(2,2,2), imshow(LH, []), title('Horizontal Detail (LH)');
subplot(2,2,3), imshow(HL, []), title('Vertical Detail (HL)');
subplot(2,2,4), imshow(HH, []), title('Diagonal Detail (HH)');

reconstructed_dwt = idwt2(LL, LH, HL, HH, 'haar');
imshow(reconstructed_dwt);
title('Reconstructed Image from DWT');



