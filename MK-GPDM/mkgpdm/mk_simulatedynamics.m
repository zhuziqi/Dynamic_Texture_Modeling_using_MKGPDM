function [X_pred, XRand_pred] = mk_simulatedynamics(X, segments, hyperpara_Kx, invKx, simSteps, simStart, kern)

% (X, segments, hyperpara_Kx, invKx, simSteps, simStart)
% invKx 表示Xin的核函数的逆

[Xin, Xout] = mk_priorIO(X, segments); % Xin 和 Xout的意义不变

Q = size(X,2);

X_pred = zeros(simSteps, Q);
XRand_pred = zeros(simSteps, Q);


order = 1;
X_pred(1,:) = simStart(:,end-Q+1:end); % X_pred(1,:)也就是X(200)
XRand_pred(1,:) = simStart(:,end-Q+1:end); % XRand_pred(1,:)也就是X(200)

for n = (order+1):simSteps % 2:simSteps
    
    RXin = Xin;
    RXout = Xout;
    invRKd = invKx;
    [X_pred(n,:), var_pred] = mk_priorManifoldOutputs([X_pred(n-1,:) zeros(1,Q)] , Xin, Xout, hyperpara_Kx, invKx, kern);
    [XRand_pred(n,:), var_pred] = mk_priorManifoldOutputs([XRand_pred(n-1,:) zeros(1,Q)] , RXin, RXout, hyperpara_Kx, invRKd, kern);   
    XRand_pred(n,:) = XRand_pred(n-1,:) + sqrt(var_pred)*randn(1,Q); % 会导致每次的结果不完全一样
end