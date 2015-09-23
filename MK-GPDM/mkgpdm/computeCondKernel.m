function [A, B] = computeCondKernel(X, hyperpara_Ky, Ky);

% ([X; X_pred], hyperpara_Ky, Ky)


n = size(Ky,1);

A = kernel(X(1:n,:), X(n+1:end,:), hyperpara_Ky);
B = kernel(X(n+1:end,:), X(n+1:end,:), hyperpara_Ky) + eye(size(X(n+1:end,:), 1))*1/hyperpara_Ky(end);


