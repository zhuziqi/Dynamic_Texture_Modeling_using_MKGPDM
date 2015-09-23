function g = mk_KernGradient(kern,X,dL_dK)

num_kerns = length(kern.comp);

g = [];

for i = 1:num_kerns
    fhandle = str2func([kern.comp{i}.type 'KernGradient']);   
    grad = fhandle(kern.comp{i},X,dL_dK);
    g = [g kern.weight(i)*grad];
end