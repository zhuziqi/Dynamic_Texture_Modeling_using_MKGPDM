function params = mk_weightsConstrain(params)

for i = 1:length(params)
    if params(i) < 0
        params(i) = 0;
    end
end

if length(find(params == 0)) < (length(params)-1)
    params = params ./ sum(params);
else
    params = rand(length(params),1);
    params = params ./ sum(params);
end


end