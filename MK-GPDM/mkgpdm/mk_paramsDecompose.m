function [X lnHyperpara_Kx lnHyperpara_Ky] = mk_paramsDecompose(params,N,Q,kern)


% 这里先把params中的X，所有的超参数都提出来    
X = reshape(params(1:N*Q), N, Q); % 把初始化的隐变量从params中抽取出来

% 把Kx的超参数提取出来，由于在这个过程中Ky是一直不变的，所以不用考虑Ky的超参数
num_kern = length(kern.comp);
num_hyperpara_Kx = 1; % 噪声项开始算
for i = 1:num_kern
    num_hyperpara_Kx = num_hyperpara_Kx + kern.comp{i}.nParams;
end

lnHyperpara_Kx = params(end-(num_hyperpara_Kx-1):end); %
lnHyperpara_Ky = params(end-(num_hyperpara_Kx+2):end-num_hyperpara_Kx); % 提取K_y的超参数，数量为3