% This is the demo script for the algorithm HOSVD proposed in [1]. Since
% the original algorithm is designed for RGB video, we use the RGB DT
% sequence for training and convert the synthesized video to GRAY for
% comparison.

% Copy the mat file from "..\Experimental Results\2. Video for Test\" for
% testing.

clear;clc;

% Add path for Tensor Toolbox V1.0
addpath('TensorToolbox_V1.0');

% Setting Parameters 
texture_name = 1;  % Choose the DT sequence for testing, set as 1,2,3,4
tau = 250; % Length of the training samples
r3 = 100; % The dimension of the subspace for the space domain 
r1 = 100; % The dimension of the subspace for the time domain 

% Load DT sequences as training samples
load(strcat(num2str(texture_name),'.mat'));
im = getfield(mov,{1,1},'cdata');
[a b c] = size(im); % Get the size of the frame

% Save the DT sequence data as a tensor
Y = zeros(a,b,tau,c);
for i = 1:tau  
    temp = double(getfield(mov,{1,i},'cdata'));    
    Y(:,:,i,1) = temp(:,:,1);
    Y(:,:,i,2) = temp(:,:,2);
    Y(:,:,i,3) = temp(:,:,3);
end
Yt = tensor(Y);

% Decompose the tensor using SVD
[Ur,~,~] = svds(double(tensor_as_matrix(Yt,1)),a);
[Vr,~,~] = svds(double(tensor_as_matrix(Yt,2)),b);
[Fr,~,~] = svds(double(tensor_as_matrix(Yt,3)),tau);
[Wr,~,~] = svds(double(tensor_as_matrix(Yt,4)),3);             

% Obtain the subspace
[I J K L] = size(Y);
Yt = tensor(Y);
r = [min(r1,a) min(r1,b) r3 3];
U = Ur(:,1:r(1));
V = Vr(:,1:r(2));
F = Fr(:,1:r(3));
W = Wr(:,1:r(4));

% Obtain the $\hat{X},\hat{A},\hat{B},\hat{V}$
Xhat = F'; 
Ahat = Xhat(:,2:tau)*pinv(Xhat(:,1:tau-1)); 
Vhat = Xhat(:,2:tau)-Ahat*Xhat(:,1:(tau-1)); 
n = size(Ahat,1);
nv = round(n/3*2);  
[Uv,Sv,Vv] = svd(Vhat,0);
Bhat = Uv(:,1:nv)*Sv(1:nv,1:nv);


% Synthesis the DT video based on 1st Order Markov Model and Linear Dynamic
% Model. 

syn_length = 500; % 500 frames are synthesized.
synth_Result(1:syn_length) = struct('frame', zeros(120, 160, 'uint8'));

% Generate new frames using the last frame of training sample as the
% initial frame
X(:,1) = Xhat(:,250);

% Calculate the dynamic trajectory of the DT sequence in the subspace
for t = 1:499 
   X(:,t+1) = Ahat*X(:,t) + Bhat*randn(nv,1);
end      

% Synthesis the first 250 frames
F = X(:,1:250)'; 
S = ttm(Yt,{U',V',F',W'});  
core = tensor(S); 
Yp = ttm(core, {U,V,F,W});
YpM1 = tensor_as_matrix(Yp,3);
for k = 1:250            
    imm_pred = reshape(YpM1(k,:),a,b,c);
    imm_pred = uint8(floor((imm_pred - floor(min(imm_pred(:))))./(ceil(max(imm_pred(:)))-floor(min(imm_pred(:))))*255));
    synth_Result(k).frame = rgb2gray(imm_pred);          
end

% Synthesis the second 250 frames
F = X(:,251:500)'; 
S = ttm(Yt,{U',V',F',W'});  
core = tensor(S);
Yp = ttm(core, {U,V,F,W});
YpM2 = tensor_as_matrix(Yp,3);

for k = 1:250            
    imm_pred = reshape(YpM2(k,:),a,b,c);
    imm_pred = uint8(floor((imm_pred - floor(min(imm_pred(:))))./(ceil(max(imm_pred(:)))-floor(min(imm_pred(:))))*255));
    synth_Result(250+k).frame = rgb2gray(imm_pred);          
end

% Output Results
save(strcat(num2str(texture_name),'_synth_Result.mat'),'synth_Result');

% Remove Path
rmpath('TensorToolbox_V1.0');