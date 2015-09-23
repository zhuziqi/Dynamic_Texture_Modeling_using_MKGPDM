function [X, hyperpara_Ky, hyperpara_Kx, w_params] = mk_gpdm(X, Y, segments, hyperpara_Ky, hyperpara_Kx, kern, options, extIters)


N = size(Y,1);  %样本数量
D = size(Y,2);  %样本维度
Q = size(X,2);  %子空间维度

lnHyperpara_Ky = log(hyperpara_Ky);   % 计算log形式的 theta， theta对应的是alpha
lnHyperpara_Kx = log(hyperpara_Kx); % 计算log形式的 thetap， thetap对应的是beta

w_options = options;
w_options(14) = 20;


for iters = 1:extIters %每一次迭代，用ME算法的话，迭代的方式可能还是需要的，
    
    fprintf(2,'Iteration %d\n',iters);
  
    % STAGE 1, OPTIMIZE THE HYPERPARAMETERS AND X

    params = [X(:)' lnHyperpara_Ky lnHyperpara_Kx];

    [params, options, flog] = scg('mk_likelihood', params, options, 'mk_gradient',Y, segments, kern);
    % 利用SCG函数对模型进行求解，得到优化的params参数
        
    % STAGE 2, OPTIMIZE THE WEIGHT PARAMETERS

    [X,lnHyperpara_Kx,lnHyperpara_Ky] = mk_paramsDecompose(params,N,Q,kern);  % 注意，在所有SCG迭代的过程中，输入输出的超参数都是ln形式的
    
    % 把Kx的超参数嵌入到结构体中，之后就一直保持不变，直到下一次的迭代    
    
    kern = mk_kernExpandParam(kern,exp(lnHyperpara_Kx));
   
    % 把结构体中的权重提取出来，作为新的params
    w_params = mk_kernWeightExtract(kern);
    
    
    [w_params, options, flog] = scg('weight_likelihood', w_params, w_options, 'weight_gradient',X, kern);
    
    w_params = mk_weightsConstrain(w_params); %  对权重做约束
    
%     plot(w_params,'-o');    
   
    kern = mk_updateKernWeight(kern,w_params); % 将权重值重新输入到核函数结构体中，进行下一次的优化
  
end

hyperpara_Ky = exp(lnHyperpara_Ky);
hyperpara_Kx = exp(lnHyperpara_Kx);