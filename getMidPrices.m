function [ midPrices ] = getMidPrices()
% Fetches mid data from observed historical prices for OIS with maturities
% 1/52, 1/12, 2/12, 3/12, 6/12, 9/12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
% 15, 20, 25 and 30 years.
    %load dataPrices.mat
    %midPrices = ((dataask(2:end,2:end) + databid(2:end,2:end))./2)'; %#ok<*COLND>
    midPrices = hdf5read('EONIAmid.hdf5','OISdataMat');
    midPrices = midPrices(:,1:500);
    size(midPrices)
end