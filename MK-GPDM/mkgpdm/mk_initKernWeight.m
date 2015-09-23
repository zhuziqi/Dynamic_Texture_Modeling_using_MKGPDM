function kern = mk_initKernWeight(kern)

% 这个函数是用来在kern结构体中添加对每个核函数的权重，这里的权重加起来应该是1

weight = (1/length(kern.comp)) * ones(length(kern.comp),1);

kern.weight = weight;

end



