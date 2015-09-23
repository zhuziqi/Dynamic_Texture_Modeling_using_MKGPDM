function hyperpara = mk_kernExtractParam(kern)

hyperpara = [];

for i = 1:length(kern.comp)
    fhandle = str2func([kern.comp{i}.type 'KernExtractParam']);  
    params = fhandle(kern.comp{i});
    hyperpara = [hyperpara params];
end

end