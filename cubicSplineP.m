function [splineCoeff, targetFunction] = cubicSplineP(t_h, delta)
% cubicSplineP(t_h, delta) 
% Insert stoptime: t_h and information decay: delta
% [splineCoeff, targetFunction]
% Returns coefficients: splineCoeff, splineCoeff(1)*x^3 +
% splineCoeff(2)*x^2 ... 
% Returns target function values: targetFunction
t = [0:t_h]';
targetFunction = exp( (t-t_h)./365 * log(delta) );
splineData = fit(t, targetFunction, 'poly3')
splineCoeff = [splineData.p1 splineData.p2 splineData.p3 splineData.p4];
end