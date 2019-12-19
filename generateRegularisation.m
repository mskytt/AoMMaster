function [H] = generateRegularisation(T_s, t_h, delta, n)
    % blkdiag for putting matrices in diagonal, with 3d matrices
    % a = A(:,:,1), for i=2:size(A,3) {a = blkdiag(a,A(:,:,i))} end
    % make spline of w²(x) and use coeffs as a^w, b^w, c^w and d^w
    % need delta, t_h, time vector (of maturities)
    
    wSplineCoeff = makeW2Spline(t_h*365, delta);

    % need to find all Q's according to definition 
    
    % finds the time index at which to change from weighted splines to
    % partly weighted/unweighted and after that unweighted
    timeIndStop = sum(T_s < t_h);
    
    % Find the matrices that makes up the splines
    if t_h > max(T_s)
        t_h = max(T_s);
        timeIndStop = timeIndStop - 1;
        warning(strcat('t_h cannot be greater than the largest value ',...
            'in T_s. t_h has been set to max(T_s).'));
    end
    [Q1,Q2] = makeQMatrices(T_s, wSplineCoeff, t_h, ...
        timeIndStop, n);
    
    I1 = zeros(4,4,timeIndStop);
    for i = 1:timeIndStop
        I1(1:2,1:2,i) = Q1(:,:,i);
    end
    I2 = zeros(4,4,n-timeIndStop+1);
    for i = 1:n-timeIndStop+1
       I2(1:2,1:2,i) = Q2(:,:,i); 
    end

    preH1 = I1(:,:,1);
    for i = 2:timeIndStop-1
        preH1 = blkdiag(preH1,I1(:,:,i));
    end
    preHmid = I1(:,:,timeIndStop) + I2(:,:,1);
    
    if timeIndStop < n
        preH2 = I2(:,:,2);
        for i = 3:n-timeIndStop+1
            preH2 = blkdiag(preH2,I2(:,:,i));
        end
        H = blkdiag(preH1,preHmid,preH2);
    else
        H = blkdiag(preH1,preHmid);
    end
end

function [wSplineCoeff] = makeW2Spline(t_h, delta)
    timevec = 0:t_h;
    w2fun = @(t) exp(log(delta)*(t-t_h)/365);
    wSplineCoeff = polyfit(timevec,w2fun(timevec),3);
end

function [Q1,Q2] = makeQMatrices(T_s, wSplineCoeff, t_h, ...
    timeIndStop, n)

        %************* Q1
        % Pre allocate parts of Q1
        Qa = zeros(2,2,timeIndStop);
        Qb = Qa;
        Qc = Qa;
        Qd = Qa;
        Q1 = Qa;
        
        % The first Sum (All parts of Q1 except the last one)
        for i = 1:timeIndStop - 1
            ti = T_s(i);
            
            Qa(:,:,i) = wSplineCoeff(1)*...
                [ 6*(T_s(i+1)^6 - T_s(i)^6)-...
                72/5*(T_s(i+1)^5 - T_s(i)^5)*ti+...
                9*(T_s(i+1)^4 - T_s(i)^4)*ti^2,...
                12/5*(T_s(i+1)^5 - T_s(i)^5)-...
                3*(T_s(i+1)^4 - T_s(i)^4)*ti;
                12/5*(T_s(i+1)^5 - T_s(i)^5)-...
                3*(T_s(i+1)^4 - T_s(i)^4)*ti,...
                T_s(i+1)^4 - T_s(i)^4 ];

            Qb(:,:,i) = wSplineCoeff(2)*...
                [ 36/5*(T_s(i+1)^5 - T_s(i)^5)-...
                18*(T_s(i+1)^4 - T_s(i)^4)*ti+...
                12*(T_s(i+1)^3 - T_s(i)^3)*ti^2,...
                3*(T_s(i+1)^4 - T_s(i)^4)-...
                4*(T_s(i+1)^3 - T_s(i)^3)*ti;
                3*(T_s(i+1)^4 - T_s(i)^4)-...
                4*(T_s(i+1)^3 - T_s(i)^3)*ti,...
                4/3*(T_s(i+1)^3 - T_s(i)^3) ];

            Qc(:,:,i) = wSplineCoeff(3)*...
                [ 9*(T_s(i+1)^4 - T_s(i)^4)-...
                24*(T_s(i+1)^3 - T_s(i)^3)*ti+...
                18*(T_s(i+1)^2 - T_s(i)^2)*ti,...
                4*(T_s(i+1)^3 - T_s(i)^3)-...
                6*(T_s(i+1)^2 - T_s(i)^2)*ti;
                4*(T_s(i+1)^3 - T_s(i)^3)-...
                6*(T_s(i+1)^2 - T_s(i)^2)*ti,...
                2*(T_s(i+1)^2 - T_s(i)^2) ];

            Qd(:,:,i) = wSplineCoeff(4)*...
                [ 12*(T_s(i+1)^3 - T_s(i)^3)-...
                36*(T_s(i+1)^2 - T_s(i)^2)*ti+...
                36*(T_s(i+1) - T_s(i))*ti^2,...
                6*(T_s(i+1)^2 - T_s(i)^2)-...
                12*(T_s(i+1) - T_s(i+1))*ti;
                6*(T_s(i+1)^2 - T_s(i)^2)-...
                12*(T_s(i+1) - T_s(i+1))*ti,...
                4*(T_s(i+1) - T_s(i)) ];
        end
        
        % The first single term (the last part of Q1)
        ti = T_s(timeIndStop);
        
        Qa(:,:,timeIndStop) = wSplineCoeff(1)*...
            [ 6*(t_h^6 - T_s(timeIndStop)^6)-...
            72/5*(t_h^5 - T_s(timeIndStop)^5)*ti+...
            9*(t_h^4 - T_s(timeIndStop)^4)*ti^2,...
            12/5*(t_h^5 - T_s(timeIndStop)^5)-...
            3*(t_h^4 - T_s(timeIndStop)^4)*ti;
            12/5*(t_h^5 - T_s(timeIndStop)^5)-...
            3*(t_h^4 - T_s(timeIndStop)^4)*ti,...
            t_h^4 - T_s(timeIndStop)^4 ];
        
        Qb(:,:,timeIndStop) = wSplineCoeff(2)*...
            [ 36/5*(t_h^5 - T_s(timeIndStop)^5)-...
            18*(t_h^4 - T_s(timeIndStop)^4)*ti+...
            12*(t_h^3 - T_s(timeIndStop)^3)*ti^2,...
            3*(t_h^4 - T_s(timeIndStop)^4)-...
            4*(t_h^3 - T_s(timeIndStop)^3)*ti;
            3*(t_h^4 - T_s(timeIndStop)^4)-...
            4*(t_h^3 - T_s(timeIndStop)^3)*ti,...
            4/3*(t_h^3 - T_s(timeIndStop)^3) ];
        
        Qc(:,:,timeIndStop) = wSplineCoeff(3)*...
            [ 9*(t_h^4 - T_s(timeIndStop)^4)-...
            24*(t_h^3 - T_s(timeIndStop)^3)*ti+...
            18*(t_h^2 - T_s(timeIndStop)^2)*ti,...
            4*(t_h^3 - T_s(timeIndStop)^3)-...
            6*(t_h^2 - T_s(timeIndStop)^2)*ti;
            4*(t_h^3 - T_s(timeIndStop)^3)-...
            6*(t_h^2 - T_s(timeIndStop)^2)*ti,...
            2*(t_h^2 - T_s(timeIndStop)^2) ];
        
        Qd(:,:,timeIndStop) = wSplineCoeff(4)*...
            [ 12*(t_h^3 - T_s(timeIndStop)^3)-...
            36*(t_h^2 - T_s(timeIndStop)^2)*ti+...
            36*(t_h - T_s(timeIndStop))*ti^2,...
            6*(t_h^2 - T_s(timeIndStop)^2)-...
            12*(t_h - T_s(timeIndStop))*ti;
            6*(t_h^2 - T_s(timeIndStop)^2)-...
            12*(t_h - T_s(timeIndStop))*ti,...
            4*(t_h - T_s(timeIndStop)) ];
        
        % Collect them in a 3D matrix for output
        for i = 1:timeIndStop
            Q1(:,:,i) = Qa(:,:,i) + Qb(:,:,i) + Qc(:,:,i) + Qd(:,:,i);
        end
        
        %************* Q2
        Q2 = zeros(2,2,n - timeIndStop+1);
        
        % The second single term
        ti = T_s(timeIndStop);
        Q2(:,:,1) = [ 12*(T_s(timeIndStop+1)^3 - t_h^3)-...
            36*(T_s(timeIndStop+1)^2 - t_h^3)*ti+...
            36*(T_s(timeIndStop+1) - t_h)*ti^2,...
            6*(T_s(timeIndStop+1)^2 - t_h^2)-...
            12*(T_s(timeIndStop+1) - t_h)*ti;
            6*(T_s(timeIndStop+1)^2 - t_h^2)-...
            12*(T_s(timeIndStop+1) - t_h)*ti,...
            4*(T_s(timeIndStop+1) - t_h) ];
        
        for i = 2:n - timeIndStop+1
            j = i + timeIndStop;
            ti = T_s(j-1);
            Q2(:,:,i) = [ 12*(T_s(j)^3 - T_s(j-1)^3)-...
                36*(T_s(j)^2 - T_s(j-1)^3)*ti+...
                36*(T_s(j) - T_s(j-1))*ti^2,...
                6*(T_s(j)^2 - T_s(j-1)^2)-...
                12*(T_s(j) - T_s(j-1))*ti;
                6*(T_s(j)^2 - T_s(j-1)^2)-...
                12*(T_s(j) - T_s(j-1))*ti,...
                4*(T_s(j) - T_s(j-1)) ];
        end
end