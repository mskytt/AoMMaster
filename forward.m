midPrices = getMidPrices();

T_k = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10]; %[0 1/52 1/12 2/12 3/12 6/12 9/12 1 2 3 4 5 6 7 8 9 10 12 15 20];% 25 30]; % Contract maturities
%T_k = [1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 2];
T_s = [0 3/12 6/12 1 2 3 4 5 6 7 8 9 10]; %25 30]; % Spline knot times
%T_s = [0, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 2];% 14/12, 16/12, 18/12, 20/12, 22/12, 2];
%T_s = [0 6/12 1 4 8 20 20 30]; % Spline knot times
n = length(T_s) - 1; % Number of spline
m = length(T_k) - 1; % Number of assets
finalMaturity = max(T_k); % Used for plotting against TtM axis
t_h = 0.5   ; % must be less than or equal to the last value in T_s to work
delta = 2;
eksde = [];     %Objective value
P = getPermutationMatrix(n); % Permutation matrix

regul = generateRegularisation(T_s, t_h, delta, n); % Computes regularisation matrix

p = 1000; % Penalty for pricing error
E = p*eye(m); % Penalty matrix, zEz 
F = eye(m);
k = 0
forwardcurves = [];
% Build curvez for all days
for day = size(midPrices,2):-1:1
 
    b_e = midPrices(1:m,day);% / 100; %zeros(m,1); % Actual prices

    f = zeros(4*n,1); % Parameters for each spline
    f_tilde = zeros(size(f)); % Initial guess
    f_bar = P*f_tilde; % Permutated initial guess (f_tilde)
    x = zeros(size(f)); % Permutated perturbations (delta_f)
    x_B = zeros(3*(n-1),1); % Basic variables
    x_N = zeros(n+3,1); % Non-basic variables
    f_bar_B = f_bar(1:length(x_B));
    f_bar_N = f_bar(length(x_B)+1:end);
    % While 
    for i = 1:5
        [B_B, B_N] = generateB(n, T_s);
        b=-B_B*f_bar_B-B_N*f_bar_N;
        H = P*regul*P';
        H_BB = H(1:length(x_B),1:length(x_B)); 
        H_BN = H(1:length(x_B),length(x_B)+1:end);
        H_NN = H(length(x_B)+1:end,length(x_B)+1:end);
        H_NB = H(length(x_B)+1:end,1:length(x_B));

        % Re-compute every iteration
        g = priceOIS(T_k, f_tilde, n, m, T_s); %zeros(m,1); % Takes forward interest to ois
        grad_g = gradientOIS(n, m, T_s, T_k, f_tilde); % Gradient of function g 
        grad_g_bar = P*grad_g; % Gradient with permutated rows
        grad_g_bar_B = grad_g_bar(1:length(x_B),:);
        grad_g_bar_N = grad_g_bar(length(x_B)+1:end,:);

        A_B = -F\grad_g_bar_B';
        A_N = -F\grad_g_bar_N';
        a = F\(g-b_e);

        H_tilde = (B_B\B_N)'*H_BB*(B_B\B_N) - 2*(B_B\B_N)'*H_BN + H_NN + ...
                  (A_N - A_B*(B_B\B_N))'*E*(A_N - A_B*(B_B\B_N));
        % Assuming b = 0
        h_tilde = (-(inv(B_B)*b)'*H_BB*(inv(B_B)*B_N)+(inv(B_B)*b)'*H_BN + ...
            (A_B*inv(B_B)*b-a)'*E*(A_N-A_B*(B_B\B_N)) + ...
            f_bar_B'*(H_BN - H_BB*(B_B\B_N)) + f_bar_N'*(H_NN - H_NB*(B_B\B_N)))';

        % Calculate solution
        x_N = -H_tilde\h_tilde;
        x_B = -B_B\B_N*x_N-b;

        timesteps = 0.001:0.001:1;
        for j = 1:length(timesteps)
            sll = timesteps(j);
            x_N_ = sll*x_N;
            x_B_ = sll*x_B;
            f_tilde_ = f_tilde + P'*[x_B ; x_N];
            f_bar_ = P*f_tilde_;
            eksde(i, j) = f_bar_'*H*f_bar_ + (A_B*x_B_+A_N*x_N_-a)'*E*(A_B*x_B_+A_N*x_N_-a);
        end

        % Update state
        f_tilde = f_tilde + P'*[x_B ; x_N];
        f_bar = P*f_tilde;
        f_bar_B = f_bar(1:length(x_B));
        f_bar_N = f_bar(length(x_B)+1:end);

        % Final result
        f = f_tilde;

        % evaluate splines

        fr = [];
        d = diff(T_s)*365;
        for i = 1:length(d)
            t = (0:d(i))/365;
            af = f((i-1)*4 + 1);
            bf = f((i-1)*4 + 2);
            cf = f((i-1)*4 + 3);
            df = f((i-1)*4 + 4);
            fr = [fr (af*(t).^3 + bf*(t).^2 + cf*(t) + df)];
        end

    end
    
    dt = finalMaturity/size(fr,2);
    pause(0.001);
    plot(dt:dt:finalMaturity,fr);
    forwardcurves(day,:) = fr;
    %display 'here last'
end
forwardCurvesSplines = flipud(forwardcurves);
% save('forwardCurveSplines.mat','forwardCurvesSplines');
% h5create('EONIA700to1500.hdf5','/MATLABFordataMat',size(forwardCurvesSplines))
% h5write('EONIA700to1500.hdf5','/MATLABFordataMat',forwardCurvesSplines);

display('saved')

