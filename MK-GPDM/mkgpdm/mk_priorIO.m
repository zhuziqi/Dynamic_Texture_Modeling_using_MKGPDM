function [Xin, Xout] = mk_priorIO(X, segments) 
% 这里使用的是一阶Markov模型，如果是高阶模型，需要重新定义

q = size(X,2);

% 提取 Xin，也就是X{1:N-1}
Xin = [zeros(1,q); X(1:end-1,:)];
Xin(segments,:) = [];

% 提取 Xout，也就是X{2:N}
Xout = X;
Xout(segments,:) = [];
