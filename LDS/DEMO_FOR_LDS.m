% This is the demo script for the algorithm LDS proposed in [1]. 

% Reference:
% [1] G. Doretto, A. Chiuso, Y. N. Wu, S. Soatto, Dynamic textures, 
% International Journal of Computer Vision 51 (2) (2003) 91-109.

% Copy the mat file from "..\Experimental Results\2. Video for Test\" for
% testing.

clear;clc;

% Setting Parameters
texture_name = 1;  % Choose the DT sequence for testing, set as 1,2,3,4
n = 50;
nv = 20;

% Load DT sequences as training samples
load(strcat(num2str(texture_name),'.mat'));
Y = zeros(120*160,length(mov));
for k = 1:length(mov)
    I = rgb2gray(mov(k).cdata);
    Y(:,k) = I(:);
end

% Suboptimal Learning of Dynamic Texture
tau = size(Y,2);
Ymean = mean(Y,2);

[U,S,V] = svd(Y-Ymean*ones(1,tau),0);

Chat = U(:,1:n);
Xhat = S(1:n,1:n)*V(:,1:n)';
Ahat = Xhat(:,2:tau)*pinv(Xhat(:,1:(tau-1)));
Vhat = Xhat(:,2:tau)-Ahat*Xhat(:,1:(tau-1));
[Uv,Sv,Vv] = svd(Vhat,0);
Bhat = Uv(:,1:nv)*Sv(1:nv,1:nv);

% Generate new frames using the last frame of training sample as the
% initial frame
X(:,1) = Xhat(:,250);
syn_length = 500; % 500 frames are synthesized.
synth_Result(1:syn_length) = struct('frame', zeros(120, 160, 'uint8'));

for t = 1:syn_length
   X(:,t+1) = Ahat*X(:,t) + Bhat*randn(nv,1);
   I = Chat*X(:,t)+Ymean;
   I = (I - floor(min(I(:))))./(ceil(max(I(:)))-floor(min(I(:))));
   synth_Result(t).frame = reshape(I,120,160);
end

% Output Results
save(strcat(num2str(texture_name),'_synth_Result.mat'),'synth_Result');