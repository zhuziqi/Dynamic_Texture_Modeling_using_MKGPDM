function [Ysample, Ymean] = mk_sampleReconstruction(X_pred, X, Y, hyperpara_Ky, Ky, invKy) 

% (X_pred, X, Y, hyperpara_Ky, Ky, invKy)


% N = size(X,1); 
M = size(X_pred,1); 
D = size(Y,2); 

% 
[A, B] = computeCondKernel([X; X_pred], hyperpara_Ky, Ky);
Km = B - A'*invKy*A; 
% mu = []; 
% C = []; 
Ymean = zeros(M,1); 
rootKm = Km^(1/2); 

z = randn(M,1);
for q = 1:D
    Ymean(:,q) = A'*invKy*Y(:,q);
%     Ysample(:,q) = (1/w(q))*rootKm*z + Ymean(:,q);
    Ysample(:,q) = rootKm*z + Ymean(:,q);
end



