% This is the demo script for the algorithm dynamic texture with fourier
% descriptors proposed in [1]. 

% Reference:
%  [1] B. Abraham, O. I. Camps, M. Sznaier, Dynamic texture with fourier
%  descriptors, in: Proceedings of the 4th nternational Workshop on Texture 
%  Analysisand Synthesis, 2005, pp. 53-58.

% Copy the mat file from "..\Experimental Results\2. Video for Test\" for
% testing.

clear;clc;

% Setting Parameters 
texture_name = 1;  % Choose the DT sequence for testing, set as 1,2,3,4
Thr = 90; 
n_fft_rgb_vec = 30;
tau = 250;  % The length of training sequence

% Load DT sequences as training samples
load(strcat(num2str(texture_name),'.mat'));
im = getfield(mov,{1,1},'cdata');
[a b c] = size(im); % Get the size of the frame

% Save the DT sequence as a array
for i = 1:tau
    temp = double(getfield(mov,{1,i},'cdata'));    
    fft_temp_gray = fft2(temp);    
    Y_fft(:,:,i) = fft_temp_gray(:);
    Y_gray(:,i) = temp(:);
end
Y_fft_gray = Y_fft(1:a*b,:);

%
thr_gray = 0;     
pp = Thr;
Mask_gray = ones(size(Y_fft_gray(:,1)));    
while(pp < nnz(Mask_gray)/numel(Mask_gray)*100 )
    thr_gray = thr_gray + 1;
    for j = 1:tau
        Mask_gray = Mask_gray & (abs(Y_fft_gray(:,j)) > thr_gray);
    end
end    
th_gray_vec = thr_gray;

% Synthesis
thr_gray = th_gray_vec(1);
Mask_gray = ones(size(Y_fft_gray(:,1)));

for i = 1:tau    
    Mask_gray = Mask_gray & (abs(Y_fft_gray(:,i)) > thr_gray);
end
pos_gray = find(Mask_gray);
L_gray = length(pos_gray);

Y_fft_masked_gray = Y_fft_gray(pos_gray,:);
Y_SVD_gray(1:L_gray,:) = real(Y_fft_masked_gray);
Y_SVD_gray(L_gray+1:2*L_gray,:) = imag(Y_fft_masked_gray);

Y_SVD_Mean = mean(Y_SVD_gray,2);
[U,S,V] = svd(Y_SVD_gray-Y_SVD_Mean*ones(1,size(Y_SVD_gray,2)),0);            

% Obtain the $\hat{X},\hat{A},\hat{B},\hat{V}$                
n_fft = n_fft_rgb_vec;     
nv_fft = round(n_fft/3*2);
first = 1:n_fft;
Chat = U(:,first); 
Xhat = S(first,first)*V(:,first)';
Ahat = Xhat(:,2:tau)*pinv(Xhat(:,1:tau-1));
Vhat = Xhat(:,2:tau)-Ahat*Xhat(:,1:(tau-1));
[Uv,Sv,Vv] = svd(Vhat,0);
Bhat = Uv(:,1:n_fft_rgb_vec)*Sv(1:n_fft_rgb_vec,1:n_fft_rgb_vec);

% Synthesis the DT video based on 1st Order Markov Model and Linear Dynamic
% Model.

% Generate new frames using the last frame of training sample as the
% initial frame
X(:,1) = Xhat(:,250);
j = sqrt(-1);

syn_length = 500; % 500 frames are synthesized.
synth_Result(1:syn_length) = struct('frame', zeros(120, 160, 'uint8'));

for t = 1:syn_length
    X(:,t+1) = Ahat*X(:,t) + Bhat*randn(n_fft_rgb_vec,1); 
    Y_res = Chat*X(:,t) + Y_SVD_Mean;

    Y_real_gray = Y_res(1:L_gray);
    Y_imag_gray = Y_res(L_gray + 1:2*L_gray);    
	Y_fft_synth = zeros(a*b,1);
    temp_gray = Y_real_gray + j*Y_imag_gray;
    Y_fft_synth(pos_gray) = temp_gray;        
    Y_synth_gray = real(ifft2(reshape(Y_fft_synth,a,b)));    
    Y_gray_synth(:,:,1) = Y_synth_gray;
    
    Y_gray_synth1 = uint8(floor((Y_gray_synth - floor(min(Y_gray_synth(:))))./(ceil(max(Y_gray_synth(:)))-floor(min(Y_gray_synth(:))))*255));
    synth_Result(t).frame = Y_gray_synth1;
end

% Output Results
save(strcat(num2str(texture_name),'_synth_Result.mat'),'synth_Result');