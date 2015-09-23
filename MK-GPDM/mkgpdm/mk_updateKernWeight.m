function kern = mk_updateKernWeight(kern,weight)

% 更新核函数的权重

num_kerns = length(kern.comp);

for i = 1:num_kerns    
    kern.weight(i) = weight(i);
end

end