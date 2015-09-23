function [dLp_dx] = mk_priorDiffX(dLp_dxin, dLp_dxout, N, Q, segments, ~)

% dLp_dxin = 
% dLp_dxout = -invKx*Xout

dLp_dx = zeros(N, Q); % 初始化为长度为N的向量

S = setdiff(1:N,mod(segments-1,N)); % 对于一阶Markov模型，S = 1:N
S(S==N) = []; % 把最后一个元素给去掉了,S = 1:N-1
dLp_dx(S,:) = dLp_dxin; % 把dLp_dxin赋值给dLp_dx中的1:N-1的位置

S = setdiff(1:N,mod(segments,N)); % S这个index调整为Xout的index，也就是2:N

dLp_dx(S,:) = dLp_dx(S,:) + dLp_dxout; % 把dLp_dx中的2:N位置的值加上了dLp_dxout

