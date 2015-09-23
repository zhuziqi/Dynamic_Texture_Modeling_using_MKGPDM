function g = testgradient(params, Y, segments, kern)

% 参数说明
% GLOBAL_PARA，用于传递全局控制参数
%
% N 观测变量的数量，也是隐变量的数量
% D 观测变量的维度
% num_hyperpara_Kx 动力学模型中的超参数个数

% params 的数据结构为 [X hyperpara_Ky hyperpara_Kx weight]

% global GLOBAL_PARA; % 系统中全局参数的结构体

recConst = 1; %GLOBAL_PARA.M_CONST;  % =1 用于调整参数beta在优化过程中损失函数中的系数，默认为1
lambda   = 1; %GLOBAL_PARA.BALANCE;  % =1，用于B-GPDM模型

N = size(Y, 1);  % 观测变量的数量，也是隐变量的数量
D = size(Y, 2);  % 观测变量的维度

% num_hyperpara_Kx = 4;  % 
% 计算动力学模型中核函数的超参数个数
num_kern = length(kern.comp);
num_hyperpara_Kx = 1; % 噪声项开始算
for i = 1:num_kern
    num_hyperpara_Kx = num_hyperpara_Kx + kern.comp{i}.nParams;
end

%% 从输入变量params中将初始化的隐变量X，K_y的超参数，K_y的超参数提取出来

% 提取隐变量X
Q = floor((length(params)-(num_hyperpara_Kx+3))/N);  % 重新提取出隐变量的维度，也就是X的维度
X = reshape(params(1:N*Q), N, Q);                    % 把数据X还原成矩阵

% 提取K_y的超参数
hyperpara_Ky = exp(params(end-(num_hyperpara_Kx+2):end-num_hyperpara_Kx)); % 提取K_y的超参数，数量为3
hyperpara_Kx = exp(params(end-(num_hyperpara_Kx-1):end));  % 提取K_X中的超参数

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

[Xin, Xout] = mk_priorIO(X, segments); % Xin和Xout分别对应于博士论文中的Xin和Xout
num_Xin = size(Xin, 1); % Xin数据的长度

%% 根据当前参数和变量计算K_y和K_x

% 计算K_y，这里使用RBF核,在本次试验中，将不对降维过程考虑多核的情况
[Ky, invKy] = computeKernel(X, hyperpara_Ky); % 计算核K_Y和K_Y^{-1}，这里使用的是RBF核加上噪声项
Ky_RBF = Ky - eye(size(X, 1))*1/hyperpara_Ky(end); % 由于GPLVM中的computeKernel函数计算的核部分包括了噪声项，这里把噪声项去掉，重新恢复成纯RBF核,用于后面求导

% 计算K_x，这里使用的是RBF核加上线性核，在我们的实验中需要把这个部分改成多核的形式
[~, sumKern]= mk_computeCompoundKernel(kern,Xin);

% sumKern = zeros(num_Xin,num_Xin);
% for i = 1:num_kern
%     sumKern = sumKern + Kx{i};
% end
sumKern = sumKern + eye(size(Xin, 1))*1/hyperpara_Kx(end);

invKx = pdinv(sumKern);

% [~, invKx, Kx_RBF] = mk_computePriorKernel(Xin, hyperpara_Kx);  % 这个是新的函数

%% 计算损失函数L相对与隐变量X，Kx的超参数hyperpara_Kx，Ky的超参数hyperpara_Ky的梯度，最后求和

% SECTION (1) 计算损失函数L对超参数hyperpara_Ky的梯度 (最后检查公式)

% 计算 dL / dKy
dL_dKy = -D/2*invKy + .5*invKy*(Y*Y')*invKy; % 这个和公式是对应的上的

% 由于在mk模型中没有为每个维度设计一个权重，所以这里不考虑w
% Yscaled = Y;
% for d=1:D
%     Yscaled(:,d) = w(d)*Y(:,d); % 对每一个维度的Y加了一个权重，也就是w(d),在我们的模型中是没有用的
% end
% dL_dK = -D/2*invKy + .5*invKy*(Yscaled*Yscaled')*invKy; % 这个和公式是对应的上的

% 计算 dKy / d hyperpara_Ky 
[dK{1}, dK{2}] = kernelDiffParams(X, X, hyperpara_Ky, Ky_RBF); % hyperpara_Ky中包含3个超参数，其中头两个是RBF核的超参数，第三个是噪声项的超参数，这里分开计算

% 利用链式法则把L对hyperpara_Ky的梯度算出来
dk = zeros(1, 3); % 应该是K_Y中超参数的倒数，因为数量是3个,需要特别说明一下的就是 dk(1)对应的是beta2的梯度，dk(2)对应的是beta1梯度
for i = 1:2
    dk(i) = sum(sum(dL_dKy.*dK{i})); % 将所有的梯度加起来
end
dk(3) = -sum(diag(dL_dKy)/hyperpara_Ky(end).^2); % 这个求得是矩阵的迹，噪声项部分的梯度

% 为了防止overfit
grad_Ky_HyperParam = dk.*hyperpara_Ky - recConst; % 对每一个参数加了一个权重项，区别于公式中除以超参数的处理方式，这里把超参数乘到前面的梯度中，如果参数越大，梯度越大的效果

% SECTION (2) 计算损失函数L对超参数hyperpara_Kx的梯度 (最后检查公式)

% 计算 dL / dKx
dL_dKx = -Q/2*invKx + 0.5*invKx*(Xout*Xout')*invKx; % -Q/2*K_x^{-1} + 0.5*K_x^{-1}*X*X^T*K_x^{-1}
dL_dKx = lambda*dL_dKx; % lambda被设定为1,用于调整动力学模型在优化中的权重

% % 计算 dKy / d hyperpara_Ky 
% dk = zeros(1, num_hyperpara_Kx);  % 初始化，求超参数的梯度，这里dk中间应该有4个参数
% [dK{1}] = lin_kernelDiffParams(Xin, hyperpara_Kx(1)); % 对线性核的参数求导
% [dK{2}, dK{3}] = kernelDiffParams(Xin, Xin, hyperpara_Kx(2:3), Kx_RBF); % 对RBF核的参数求导
% 
% % 利用链式法则把L对hyperpara_Kx的梯度算出来
% for i = 1:(num_hyperpara_Kx-1)
%     dk(i) = sum(sum(dL_dKx.*dK{i})); % dL_dK_x * dK_x/d\lambda
% end
% dk(num_hyperpara_Kx) = -sum(diag(dL_dKx)/hyperpara_Kx(end).^2); % 噪声项的梯度

% 计算 dKy / d hyperpara_Ky, 并利用链式法则把L对hyperpara_Kx的梯度算出来
dk = mk_KernGradient(kern,Xin,dL_dKx);
dk(num_hyperpara_Kx) = -sum(diag(dL_dKx)/hyperpara_Kx(end).^2); % 噪声项的梯度

% 为了防止overfit
grad_Kx_HyperParam = dk.*hyperpara_Kx - 1; % 对每一个参数加了一个权重项，区别于公式中除以超参数的处理方式，这里把超参数乘到前面的梯度中，如果参数越大，梯度越大的效果

% SECTION (3) 计算损失函数L对隐变量X的梯度，包括三个部分，分别是dKy，dKx和Xout (最后检查公式)

% 计算L中dKy部分对X的梯度，利用链式法则，计算dL / dKy * dKy / dX
dL_dx = zeros(N, Q);  % 似然函数对隐变量模型求偏导，
for d = 1:Q
    Kpart = kernelDiffX(X, hyperpara_Ky, d, Ky_RBF); % 直接是dK_y/dx的函数，但是需要每一个维度的求
    dL_dx(:, d) = 2*sum(dL_dKy.*Kpart, 2) - diag(dL_dKy).*diag(Kpart); % dL_dx = dL/dK * dK_y/dx 但是这里减去对角阵不知道什么意思,在试验中，Kpart的所有对焦元素全部都是0，所以这里也就没有减去任何值
end

% 计算L中dKx部分对Xin的梯度，利用链式法则，计算dL / dKx * dKx / dxin (维度N-1),这里修改了
dL_dxin = zeros(num_Xin, Q); % 损失函数对Xin求导
% for d = 1:Q % 每一个维度进行计算
% 	Kpart = lin_kernelDiffX(Xin, hyperpara_Kx(1), d) + kernelDiffX(Xin, hyperpara_Kx(2:3), d, Kx_RBF); % dK_X/d_xin
%     dL_dxin(:, d) = 2*sum(dL_dKx.*Kpart, 2) - diag(dL_dKx).*diag(Kpart); % dK_X/d_xin
% end

gX = mk_KernGradX(kern,Xin);

for d = 1:Q % 每一个维度进行计算
    gX_temp(:,:) = gX(:,d,:);
    dL_dxin(:, d) = 2*sum(dL_dKx.*gX_temp', 2);
end

% 计算L中Xout部分对X的梯度，dL / dXout = 1/2 (Kx^{-1}*Xout + Kx^{-T}*Xout),然后按照维度把
% dL/dXout + dL/dXin 合并到一起，形成一个长度为N维的梯度向量，dL / dXout = -lambda*invKx*Xout
dLp_dx = mk_priorDiffX(dL_dxin, -lambda*invKx*Xout, N, Q, segments); % 简单的说，做了一件事情，就是把 dLp_dxin + dLp_dxout，加的时候按照对应的位置，这里的dLp_dxout设定为-lambda*invKp*Xout

dLp_dx(segments,:) = dLp_dx(segments,:) - lambda*X(segments,:); % dLp_dx{1} = dLp_dx{1}减去初始化X的t=1位置的值，这个步骤的意义不清楚

%
dL_dx = dL_dx + dLp_dx; % dK_Y/d_xin + dK_X/d_xin

gX= dL_dx(:)';

%% 最后把所有的梯度汇总到一起
g = -[gX(:)' grad_Ky_HyperParam grad_Kx_HyperParam]; % g中间包含了对隐变量求导的数值结果，两个超参数的数值结果