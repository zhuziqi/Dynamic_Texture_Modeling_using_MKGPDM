function [K, invK, rbfK] = mk_computePriorKernel(X, theta)

% 这个是GPDM的新模型，用来计算对K_X的核函数

rbfK = kernel(X, X, theta(2:3));
K = lin_kernel(X, X, theta(1)) + rbfK + eye(size(X, 1))*1/theta(end);
invK = pdinv(K);
