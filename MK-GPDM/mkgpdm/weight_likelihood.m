function L = weight_likelihood(params, X, kern)

% 在这里 params 就是所有的权重信息了，也就是params = [w1 w2 ... wn]
segments = 1; %这里仅考虑一阶markov模型
% 方案1，在每一次计算之前，对权重参数进行约束，
params = mk_weightsConstrain(params);

Q = size(X, 2); % 变量的长度
[Xin, Xout] = mk_priorIO(X, segments); % 把Xin和Xout从X中间抽取出来，这里仅考虑一阶Markov模型的情况

% 首先将params中间的参数更新到kern结构体中
kern = mk_updateKernWeight(kern,params);

% 计算带有权重的核函数

[~, sum_kerns] = mk_computeCompoundKernel(kern,Xin);

invK = pdinv(sum_kerns);

L = Q/2 * logdet(sum_kerns) + 1/2 * trace(invK * (Xout * Xout')) + norm(params);