function [g] = priceOIS(T_k, f, n, m, T_s)

% Numerator
T = zeros(m,1);
for c = 1:m
    T(c) = 1 - exp(-spotRate(T_k(c+1), f, n, T_s) * T_k(c+1));
end

% Denominator
N = zeros(m,1);
for c = 1:m
    if T_k(c+1) < 1
        N(c) = T_k(c+1)*exp(-spotRate(T_k(c+1), f, n, T_s) * T_k(c+1));
    else
        for t = 1:T_k(c+1)
            N(c) = N(c) + exp(-spotRate(t, f, n, T_s) * t);
        end
    end
end
g = T./N;
end