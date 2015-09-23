function gX = mk_KernGradX(kern,X)

num_kerns = length(kern.comp);

[N,Q] = size(X);

gX = zeros(N,Q,N);

for i = 1:num_kerns    
    fhandle = str2func([kern.comp{i}.type 'KernGradX']);    
    gX = gX + kern.weight(i)*fhandle(kern.comp{i}, X, X);  
end

end