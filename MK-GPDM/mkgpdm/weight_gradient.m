function g = weight_gradient(params, X, kern)

% 在这里 params 就是所有的权重信息了，也就是params = [w1 w2 ... wn]

segments = 1; %这里仅考虑一阶markov模型
% 方案1，在每一次计算之前，对权重参数进行约束，
params = mk_weightsConstrain(params);

[Xin, Xout] = mk_priorIO(X, segments); % 把Xin和Xout从X中间抽取出来，这里仅考虑一阶Markov模型的情况
Q = size(Xout, 2); % 变量的长度

% 首先将params中间的参数更新到kern结构体中
kern = mk_updateKernWeight(kern,params);

[Kx, sum_Kx] = mk_computeCompoundKernel(kern,Xin);

invKx = pdinv(sum_Kx);

% 现在开始计算梯度项，梯度项由三个部分构成，分别是Q/2 * ln det(K), 1/2 * tr(K^-1 * Xout *
% Xout^T)，以及正则项,注意，在这里，最后梯度应该是一个矩阵，对于每一个需要求梯度的变量，应该有一个梯度值与之对应

% 先计算dL/dKx

dL_dKx = -Q/2*invKx + 0.5*invKx*(Xout*Xout')*invKx;

% dKx/dw 其实就是每一个Kx，所以
g = zeros(length(kern.comp),1);

for i = 1:length(kern.comp)
    g(i) = sum(sum(dL_dKx .* Kx{i}));
end

% 正则项对于每一个权重参数的梯度

norm2 = 1/norm(params,2); % 计算(w1^2+w2^2+...+wn^2)^{-0.5}

for i = 1:length(kern.comp)
    g(i) = g(i) + norm2 * params(i);
end

% 为了避免梯度太大，这里保持梯度的方向不变，对梯度的长度进行归一化，也就是间接的减小优化步长
g = g/norm(g);

g = -g';