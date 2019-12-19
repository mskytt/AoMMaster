function [B_B, B_N] = generateB(n, T_s)
    B = zeros(3*(n-1),4*n);

    subBlockB = zeros(3,4);
    subBlockFunction = @(T_2i,T_i) [3*(T_2i-T_i),       1,        0,          0;
                                    3*(T_2i-T_i)^2,  2*(T_2i-T_i), 1,          0;
                                    (T_2i-T_i)^3,   (T_2i-T_i)^2, (T_2i-T_i), 1];

    zeroNegEye = [zeros(3,1) -eye(3)];

    for t = 1:n-1
        B(3*t-2:3*t,4*t-3:4*t+4) = [subBlockFunction(T_s(t+1),T_s(t)), zeroNegEye];
    end

    B_mod = B;
    for i = 4:size(B_mod,1)
        for j = 0:mod(i-4,3)
            multby2 = ((mod(i-5,3) == 0) && j == 1);
            t_i = floor((i-1)/3); % for row 4 t_i = 1
            B_mod(i,:) = B_mod(i,:) + B_mod(i-3-j,:)*(T_s(t_i+2) - T_s(t_i+1)).^j *(1 + multby2);
        end
    end

    B_N = B_mod(:,[1:5,9:4:end]);
    % extract column indeces used for B_B matrix
    indexesBB = 6:size(B_mod,2);
    indexUnWanted = 4:4:length(indexesBB);
    indexesBB(indexUnWanted) = [];
    B_B = B_mod(:,indexesBB);

end