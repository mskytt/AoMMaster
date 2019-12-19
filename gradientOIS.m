function [dG] = gradientOIS(n, m, T_s, T_k, f)

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

% d-numerator
dT = zeros(4*n, m);
for c = 1:m
    maturity = T_k(c+1);
    dot = 1*exp(-spotRate(maturity, f, n, T_s) * maturity);
    d = 1;
    for s = 1:n
        if maturity > T_s(s+1)
            dT(d,c) = dot*(1/4)*(T_s(s+1)-T_s(s))^4;
            dT(d+1,c) = dot*(1/3)*(T_s(s+1)-T_s(s))^3; 
            dT(d+2,c) = dot*(1/2)*(T_s(s+1)-T_s(s))^2;
            dT(d+3,c) = dot*(T_s(s+1)-T_s(s));
            d = d + 4;
        else 
            dT(d,c) = dot*(1/4)*(maturity-T_s(s))^4;
            dT(d+1,c) = dot*(1/3)*(maturity-T_s(s))^3; 
            dT(d+2,c) = dot*(1/2)*(maturity-T_s(s))^2;
            dT(d+3,c) = dot*(maturity-T_s(s));
            d = d + 4;     
            break;
        end
    end
    dT(d:end,c) = 0;
end

% d-denominator
dN = zeros(4*n, m);
for c = 1:m
    t = 1:T_k(c+1);
    if isempty(t)
        t = T_k(c+1);
    end
    delta = diff([0 t]);
    
    d = 1;
    for s = 1:n
        ind = find(t >= T_s(s));
        for i=ind
            r = spotRate(t(i), f, n, T_s);
            deltaT = delta(i);
            tau = min(T_s(s+1), t(i));
            
            dN(d,c)   = dN(d,c)   + deltaT*(-1)*exp(-r*t(i))*(1/4)*(tau - T_s(s))^4;
            dN(d+1,c) = dN(d+1,c) + deltaT*(-1)*exp(-r*t(i))*(1/3)*(tau - T_s(s))^3;
            dN(d+2,c) = dN(d+2,c) + deltaT*(-1)*exp(-r*t(i))*(1/2)*(tau - T_s(s))^2;
            dN(d+3,c) = dN(d+3,c) + deltaT*(-1)*exp(-r*t(i))*      (tau - T_s(s));
        end
        
        d = d + 4;
    end
end

% Gradient
dG = (dT.*repmat(N', 4*n, 1) - dN.*repmat(T', 4*n, 1)) ./ (repmat(N', 4*n, 1).^2);
end