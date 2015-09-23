function [Ysample, Ymean] = mk_prediction(X, Y, hyperpara_Ky, hyperpara_Kx, weight, kern, predFrames, segments)

% 计算Ky和invKy
[Ky, invKy] = computeKernel(X, hyperpara_Ky);

% 计算Xin
[Xin, ~] = mk_priorIO(X, segments);

% [~, invKd] = mk_computePriorKernel(Xin, hyperpara_Kx);

% 把和Kx相关的超参数和权重都嵌入到核结构体中
kern = mk_kernExpandParam(kern,hyperpara_Kx(1:(end-1)));
kern = mk_updateKernWeight(kern,weight);

% 计算Kx
[~, sum_kerns] = mk_computeCompoundKernel(kern, Xin);
Kx = sum_kerns + eye(size(Xin, 1))*1/hyperpara_Kx(end); % 多核然后加上噪声项

% 计算invKx
invKx = pdinv(Kx);

% 生成帧的数量
simSteps = predFrames;

% starts at ened of training sequence;
% 在使用二阶Markov模型时，这一步意义不明确
simStart = [X(segments(1)+1,:) X(end,:)]; %  inputs 2 points in case using 2nd order model 这里simStart就是[X(2) X(200)]

% 生成的simSteps帧隐变量
[X_pred, ~] = mk_simulatedynamics(X, segments, hyperpara_Kx, invKx, simSteps, simStart, kern);

% 再把隐变量重构到高维空间
[Ysample, Ymean] = mk_sampleReconstruction(X_pred, X, Y, hyperpara_Ky, Ky, invKy);
