function [ P ] = getPermutationMatrix( n )
% Generate the permutation matrix
    
    P = zeros(4*n);
    xn_length = 3+n;
    xb_length = 4*n-xn_length;
    
    P(xb_length+1:xb_length+4,1:4) = eye(4);
    
    rowCounter = -1;
    for i = 5:4*n
        if mod(i-1,4)==0
           P(xb_length+(i-1)/4+4,i) = 1;
           rowCounter = rowCounter + 1;
        else
            P(i-5-rowCounter,i) = 1;
        end
    end
    
end