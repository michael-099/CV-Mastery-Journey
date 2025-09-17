pkg load image;

img_dir    = fullfile(pwd, 'faces');
img_size   = [64, 64];
min_images = 8;
top_k_display = 5;
k_recon = 20;

exts = {'*.pgm','*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'};
img_files = [];

if exist(img_dir, 'dir')
    for e = 1:numel(exts)
        f = dir(fullfile(img_dir, exts{e}));
        if ~isempty(f)
            img_files = [img_files; f];
        end
    end
end

if isempty(img_files)
    fprintf('Folder "%s" missing or empty. Using synthetic fallback dataset.\n', img_dir);
    base = load_image();
    if ndims(base) == 3, base = rgb2gray(base); end
    base = im2double(imresize(base, img_size));
    data_imgs = cell(1, min_images);
    for i = 1:min_images
        I = base;
        switch mod(i-1,6)
            case 1, I = fliplr(I);
            case 2, I = flipud(I);
            case 3, I = imrotate(I, 15, 'bilinear', 'crop');
            case 4, I = imrotate(I, -15, 'bilinear', 'crop');
            case 5, I = imadjust(I, stretchlim(I,0.02));
        end
        I = imnoise(I, 'gaussian', 0, 0.001 * i);
        data_imgs{i} = im2double(imresize(I, img_size));
    end
    num_imgs = numel(data_imgs);
    data = zeros(prod(img_size), num_imgs);
    for i = 1:num_imgs
        data(:,i) = data_imgs{i}(:);
    end
else
    num_imgs = numel(img_files);
    data = zeros(prod(img_size), num_imgs);
    for i = 1:num_imgs
        img = imread(fullfile(img_files(i).folder, img_files(i).name));
        if ndims(img) == 3, img = rgb2gray(img); end
        img = im2double(imresize(img, img_size));
        data(:, i) = img(:);
    end
end


mean_face = mean(data, 2);
centered_data = data - mean_face;

C = centered_data' * centered_data;
[Vec, Val] = eig(C);
eigvals = diag(Val);
[eigvals_sorted, idx] = sort(eigvals, 'descend');
Vec = Vec(:, idx);
eigenfaces = centered_data * Vec;
eigenfaces = bsxfun(@rdivide, eigenfaces, sqrt(sum(eigenfaces.^2,1)) + eps);

figure('Name','Eigenfaces - Mean and Reconstructions','NumberTitle','off','Position',[100 100 1200 600]);
subplot(2, top_k_display+1, 1);
imshow(reshape(mean_face, img_size), []);
title('Mean Face');

for i = 1:top_k_display
    subplot(2, top_k_display+1, 1+i);
    imshow(reshape(eigenfaces(:, i), img_size), []);
    title(sprintf('Eigenface %d', i));
end

k = min(k_recon, size(eigenfaces,2));
Uk = eigenfaces(:, 1:k);

for j = 1:min(5, num_imgs)
    proj = Uk' * (centered_data(:,j));
    recon = Uk * proj + mean_face;
    subplot(2, top_k_display+1, top_k_display+1+j);
    imshow(reshape(recon, img_size), []);
    if j == 1
        title(sprintf('Reconstructed (k=%d)', k));
    else
        title(sprintf('Recon %d', j));
    end
end

fprintf('PCA completed: %d images, size %dx%d, computed %d eigenfaces.\n', ...
    num_imgs, img_size(1), img_size(2), size(eigenfaces,2));

