function L = testlikelihood(params, Y, segments, kern)


% global W_VARIANCE
% global M_CONST
% global BALANCE


% recConst = M_CONST; % =1 用于调整参数beta在优化过程中损失函数中的系数，默认为1
% lambda = BALANCE;   % =1，用于B-GPDM模型

N = size(Y, 1); % 观测变量的长度
D = size(Y, 2); % 观测变量的维度

% num_hyperpara_Kx = 4; % 动力学模型中超参数的个数

num_kern = length(kern.comp);
num_hyperpara_Kx = 1; % 噪声项开始算
for i = 1:num_kern
    num_hyperpara_Kx = num_hyperpara_Kx + kern.comp{i}.nParams;
end

%% 从输入变量params中将初始化的隐变量X，K_y的超参数，K_y的超参数提取出来

% 提取隐变量X
Q = floor((length(params)-(num_hyperpara_Kx+3))/N);  %重构回来隐变量X的维度，也就是降维后的维度
X = reshape(params(1:N*Q), N, Q); % 把初始化的隐变量从params中抽取出来

% 提取K_y的超参数
hyperpara_Ky = exp(params(end-(num_hyperpara_Kx+2):end-num_hyperpara_Kx)); % 提取K_y的超参数，数量为3
hyperpara_Kx = exp(params(end-(num_hyperpara_Kx-1):end)); % 提取K_X中的超参数，这里需要变化

% fprintf( 'hyperpara_Ky: %s \n',num2str(hyperpara_Ky));
% fprintf( 'hyperpara_Kx: %s \n',num2str(hyperpara_Kx));
% 将超参数嵌入到结构体中

kern = mk_kernExpandParam(kern,hyperpara_Kx);

% init_para = 1;
% for i = 1:length(kern.comp)
%     fhandle = str2func([kern.comp{i}.type 'KernExpandParam']);  
% 	kern.comp{i} = fhandle(kern.comp{i}, hyperpara_Kx(init_para:init_para+kern.comp{i}.nParams-1)); % 从外界向结构体中输入超参数
%     init_para = init_para + kern.comp{i}.nParams;
% end
% dynamic_noise_para = exp(params(end));

% theta = thetaConstrain(theta);
% thetap = thetaConstrain(thetap);

%% 根据Markov模型将Xin和Xout分出来，这里不考虑样本中有缺失的情况
[Xin, Xout] = mk_priorIO(X, segments); % 把Xin和Xout从X中间抽取出来，这里仅考虑一阶Markov模型的情况
num_Xin = size(Xin, 1);  % Xin数据的长度

%% 根据当前参数和变量计算K_y和K_x
[Ky, invKy] = computeKernel(X, hyperpara_Ky); % 根据X，计算K_Y，使用的是RBF核
% [Kx, invKx, ~] = mk_computePriorKernel(Xin, hyperpara_Kx);  % 这个是新的函数

[~, sum_kerns] = mk_computeCompoundKernel(kern, Xin);

% Kx = zeros(num_Xin,num_Xin);
% for i = 1:num_kern
%     Kx = Kx + mk_kern{i};
% end
Kx = sum_kerns + eye(size(Xin, 1))*1/hyperpara_Kx(end); % 多核然后加上噪声项
invKx = pdinv(Kx);
clear mk_kern;


%% 计算损失函数L

LOG2PI = log(2*pi); % 为了节省计算时间，做了一个预运算

% Constant = - (DN+Q(N-1))/2 * ln 2pi 
CONST = -D*N/2*LOG2PI - Q*num_Xin/2*LOG2PI; % 损失函数中的常数项，num_Xin = N-1

% L_part1 = - D/2 * log |Ky|
L_part1 = -D/2*logdet(Ky); 

% L_part2 = 1/2 * tr(Ky^{-1}*Y*Y^T)
L_part2 = 0;
for d= 1:D % L = L - sum_i^D (-1/2 * w_i * w_i * Y_i^T * K^-1 * Y_i)
    L_part2 = L_part2 -0.5*Y(:, d)'*invKy*Y(:, d);
end

% L_part3 = Q/2 * log |Kx|
L_part3 = - Q/2*logdet(Kx);

% L_part4 = 1/2 * tr(Ky^{-1}*Xout*Xout^T)
L_part4 = 0;
for d= 1:Q
    L_part4 = L_part4 - 0.5*Xout(:, d)'*invKx*Xout(:, d); % -0.5 * SUM_i (Xout_i^T * K_x^{-1} * Xout_i) 
end

% L_part5 = sum(ln(hyperpara_Ky)) + sum(ln(hyperpara_Kx))
L_part5 = - sum(log(hyperpara_Ky)) - sum(log(hyperpara_Kx));

% L_part6 = ln p(x_1)
L_part6 = - size(segments,2)*Q/2*LOG2PI - 0.5*sum(sum(X(segments,:).*X(segments,:)));

% 到这里其实就全部完整了 
L = CONST + L_part1 + L_part2 + L_part3 + L_part4 + L_part5 + L_part6;

% W_part 
% W_part = N*sum(log(w)); % L = L + N*sum(log w_i)
% if (W_VARIANCE > 0) 
%     W_part = W_part + D*log(2) - D/2*log(2*pi*W_VARIANCE) - 0.5/W_VARIANCE*sum(w.*w) ; % L = L + D * log(2) - *** 后面的部分和权重参数w相关
% end
% L = -(L + W_part);

L = -L;