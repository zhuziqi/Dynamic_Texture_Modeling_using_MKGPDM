function kern = mk_creatkern(X,kernType)

% 应该是可以直接使用复合核的结构，但是现在还不会用

% kernType = {'rbf','lin','poly'};

num_kern = length(kernType);

kern = cell(num_kern,1);

for i=1:num_kern
    creat_kern = kernCreate(X,kernType);    
    kern{i} = creat_kern;
end


end