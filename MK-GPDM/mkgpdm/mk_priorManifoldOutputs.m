function [xo, varxo] = mk_priorManifoldOutputs(xi, Xin, Xout, ~, invKx, kern)

% Xin, Xout, hyperpara_Kx, invKx
% xi 后面Q个维度全部是0
Q = size(Xout,2);

input = [xi(:,1:Q)]; % [X_pred(n-1,:) zeros(1,Q)], input = X_pred(n-1,:) 当n=2的时候，也就是X(200),往后就是X 

N = size(Xout, 1); % Xin的维度
M = size(input, 1); % 输入只有一个，也就是x(t)

%　alpha = zeros(N, Q);
% 这里要修改成多核的形式
% kbold = lin_kernel(input, Xin, hyperpara_Kx(1))' + kernel(input, Xin, hyperpara_Kx(2:3))'; % 线性核加上高斯核

[~, kbold] = mk_computeCompoundKernel(kern,input,Xin); % 计算
kbold = kbold';

% 计算\mu_x 也就是均值
A = Xout'*invKx; % Xout^T * Kx^-1
output = A*kbold; % Xout^T * Kx^-1 * Kx(X)

output = output'; % 这一步估计是为了调整数据的格式

% 计算\sigma_x 也就是方差
output_var = zeros(M, 1);
% 这个到底是什么意思呢？
for i = 1:M
    [~, kxx] = mk_computeCompoundKernel(kern,input(i,:),input(i,:));
    output_var(i) = kxx - kbold(:, i)'*invKx*kbold(:, i);
%    output_var(i) = lin_kernel(input(i,:), input(i,:), hyperpara_Kx(1)) + hyperpara_Kx(3) +1/hyperpara_Kx(end) - kbold(:, i)'*invKx*kbold(:, i); 
end

xo = output;
varxo = output_var;

