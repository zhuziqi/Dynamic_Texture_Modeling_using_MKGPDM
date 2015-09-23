function [K1, K2] = kernelDiffParams(arg1, arg2, arg3, arg4);

% KERNELDIFFPARAMS Get gradients of kernel wrt its parameters.

if nargin < 3
  X1 = arg1;
  X2 = arg1;
  theta = arg2;
else
  X1 = arg1;
  X2 = arg2;
  theta = arg3; % 就是核函数的三个参数
end

theta = thetaConstrain(theta);

if nargin < 4
    K = kernel(X1, X2, theta);
else
    K = arg4; % 核函数，这里输入的核函数对角线上的元素都是1
end
K2 = K/theta(2); % 在这个模型里面，theta(2)就是论文里面的beta1，theta(1)就是beta2, dK_y/d \beta_1
K1 = -0.5*dist2(X1, X2).*K; % 这里对应的就是论文里面dK_y/d \beta_2

