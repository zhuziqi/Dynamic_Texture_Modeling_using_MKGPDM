function kx = kernel(x, xKern, theta)

% KERNEL Compute the rbf kernel

n2 = dist2(x, xKern);  % 计算向量X和向量xKern之间的平方欧式距离，如果输入是M*N的，那么输出就是M*M的，分别计算每一个元素的距离
wi2 = theta(1)/2;      
kx = theta(2)*exp(-n2*wi2);
   