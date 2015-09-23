function [K, sum_kerns] = mk_computeCompoundKernel(kern,X1,X2)

% 这个函数是用来计算多核函数的结果
% kern结构体，使用的是GP-MAT代码中的结构体，但是没有使用复合结构体，主要是对复合结构体中的嵌套结构不是很清楚
% X，数据变量
% sum_kerns 带权重的多核函数的和
% K 不带权重的每个核函数计算结果，用在多核学习的过程中

% kernelType = {'rbfKern','linKern'};

num_kerns = length(kern.comp);
K  = cell(num_kerns,1);

if (nargin == 2) % 计算自相关核函数

    sum_kerns = zeros(size(X1,1),size(X1,1));

    for i = 1:num_kerns    
        fhandle = str2func([kern.comp{i}.type 'KernCompute']);    
        K{i} = fhandle(kern.comp{i}, X1, X1);
        sum_kerns = sum_kerns + kern.weight(i) * K{i};
    end
elseif (nargin == 3) % 计算自相关核函数

    sum_kerns = zeros(size(X1,1),size(X2,1));
    for i = 1:num_kerns    
        fhandle = str2func([kern.comp{i}.type 'KernCompute']);    
        K{i} = fhandle(kern.comp{i}, X1, X2);
        sum_kerns = sum_kerns + kern.weight(i) * K{i};
    end
    
end