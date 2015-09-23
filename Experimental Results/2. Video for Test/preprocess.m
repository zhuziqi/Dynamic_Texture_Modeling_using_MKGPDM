% Version 0.1, pubished on Sep 21, 2015
% Contributed by Ziqi Zhu

% Video Preprocess

% This script is used to pre-process the orginial dynamic texture videos
% and generate the M file for modeling algorithms. Due to the limitation of
% Memory, all the testing videos are resized into 160 \times 120. For each
% DT sequence, we use 250 frames for training. For the convenience of
% programming, we rename each DT sequence as:
% seawave.avi  -> 1.avi
% straw.avi    -> 2.avi
% actinia.avi  -> 3.avi
% sunshade.avi -> 4.avi

% Parameters
im_width = 160;
im_height = 120;

for NUM = 1:4
    texture_name = num2str(NUM);
    name = [texture_name,'.avi'];

    input_video = VideoReader(name);
    mov(1:250) = struct('cdata', zeros(im_height, im_width, 3, 'uint8'));

    for i =1:250
        I = read(input_video, i);
        for j = 1:3
            mov(i).cdata(:,:,j) = imresize(I(:,:,j), [im_height,im_width]);
        end
    end
    save(strcat(num2str(NUM),'.mat'),'mov');
end


