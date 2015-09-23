% This is the demo script for the algorithm GPDM [1] used for 
% Dynamic Texture synthesis

% Reference:
% [1] J. M. Wang, D. J. Fleet, A. Hertzmann, Gaussian process dynamical 
% models for human motion, Pattern Analysis and Machine Intelligence, IEEE
% Transactions on 30 (2) (2008) 283–298.

% In this script, the netlab toolbox and some source code from gplvm is
% used.

% Copy the mat file from "..\Experimental Results\2. Video for Test\" for
% testing.

clear;clc;

% Add path for Tensor Toolbox V1.0
addpath('netlab');
addpath('gplvm');
addpath('gpdm');
addpath('util');
% Setting Parameters 
texture_name = 4;  % Choose the DT sequence for testing, set as 1,2,3,4
tau = 250;  % The length of training sequence

% Load image data
load(strcat(num2str(texture_name),'.mat'));
im = getfield(mov,{1,1},'cdata');
[a b c] = size(im); % Get the size of the frame

Y = zeros(tau,a*b);

for i = 1:tau
    I = double(rgb2gray(getfield(mov,{1,i},'cdata')));
    Y(i,:) = I(:);
end

%% Set GPDM parameters
format long

global USE_GAMMA_PRIOR  % gamma prior for dynamics, only works with RBF kernel
global GAMMA_ALPHA % defines shape of the gamma prior
global USE_LAWRENCE % fix dynamics HPs, as Lawrence suggested (use with thetad = [0.2 0.01 1e6];) 
global FIX_HP % fix all HPs
global MARGINAL_W % marginalize over W while learning X
global MARGINAL_DW % marginalize over scale in dynamics while learning X
global LEARN_SCALE % use different scales for different output dimensions
global REMOVE_REDUNDANT_SCALE % let W absorb the overall scale of reconstruction
global W_VARIANCE % kappa^2 in the paper, not really the variance though
global M_CONST % M value in Jack's master's thesis
global BALANCE % Constant in front of dynamics term, set to D/q for the B-GPDM
global SUBSET_SIZE % Number of data to select for EM, set -1 for all data. 
global USE_OLD_MISSING_DATA

M_CONST = 1; 
REMOVE_REDUNDANT_SCALE = 1;
LEARN_SCALE = 1; 
MARGINAL_W = 0; 
MARGINAL_DW = 0; 
W_VARIANCE = 1e6; 
FIX_HP = 0; 
USE_GAMMA_PRIOR = 0; 
GAMMA_ALPHA = [5 10 2.5]; 
USE_LAWRENCE = 0;
BALANCE = 1;
SUBSET_SIZE = -1; 

opt = foptions;
opt(1) = 1;
opt(9) = 0;
if MARGINAL_W == 1
    opt(14) = 100; % total number of iterations
    extItr = 1; 
else
    opt(14) = 20; % rescaling every 10 iterations
    extItr = 200; % do extItr*opt(14) iterations in total
end  
modelType = [2 0 5]; 
missing = [];
N = size(Y, 1); 
D = size(Y, 2);  
q = 3; % dimensionality of latent space

%% Initialize the latent variable using PCA
X = zeros(N, q);
meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); % Remove the bais of data
clear im mov 

[v, u] = pca(Y); % Large memory is required by this step. 
v(v<0)=0;
X = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); % 保留3维向量，并且进行标准差的标准化

%% Initialize hyperparameters
segments = 1;

theta = [1 1 exp(1)];
thetad = [0.9 1 0.1 exp(1)];
w = ones(D,1);

%% Optimization
[X theta thetad w] = gpdmfitFull(X, Y, w, segments, theta, thetad, opt, extItr, modelType, missing);

save example_model X Y w theta thetad modelType N D q meanData segments initY varY missing;
clear;clc; % Remove redundant variable and save memory
%% Synthesize sample sequence from learned model

load example_model
[K invK] = computeKernel(X, theta);
[Xin Xout] = priorIO(X, segments, modelType);
[Kd invKd] = computePriorKernel(Xin, thetad, modelType(3));
simSteps = 500; % 500 frames are synthesized.
% starts at ened of training sequence;
simStart = [X(segments(1)+1,:) X(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred XRand_pred] = simulatedynamics(X, segments, thetad, invKd, simSteps, simStart, modelType);

[Ysample Ymean] = sampleReconstruction(X_pred, X, Y, theta, w, K, invK) ;

Ysample1 = real(Ysample);

%% Output Results
synth_Result(1:simSteps) = struct('frame', zeros(120, 160, 'uint8'));

for t = 1:simSteps
   I = Ysample1(t,:);
   I = (I - floor(min(I(:))))./(ceil(max(I(:)))-floor(min(I(:))));
   synth_Result(t).frame = reshape(I,120,160);
end
save(strcat(num2str(texture_name),'_synth_Result.mat'),'synth_Result');

rmpath('netlab');
rmpath('gplvm');
rmpath('gpdm');
rmpath('util');