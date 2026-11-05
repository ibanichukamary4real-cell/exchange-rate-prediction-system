%% EXCHANGE RATE PREDICTION USING NEURAL NETWORK
% Main program of the Exchange Rate Prediction System.
%
% Author : Lobarine Emmanuel Kinane
% Purpose: Predicts the Naira exchange rate against the United States Dollar
%          (USD), Euro (EUR), British Pound Sterling (GBP) and Canadian
%          Dollar (CAD) with a multilayer perceptron (4 inputs, 10 hidden
%          neurons, 4 outputs) trained with the Levenberg-Marquardt algorithm.
%
% How it works:
%   1. Loads annual average rates from data/exchange_rates.csv.
%   2. Uses the rates of year t as inputs and the rates of year t+1 as targets.
%   3. Divides the samples into 60% training, 20% validation, 20% test.
%   4. Trains the network with early stopping on the validation set.
%   5. Reports MSE, RMSE and MAPE on the test set.
%   6. Predicts the rates for the year after the last year in the data.
%   7. Compares the prediction with a first-degree polynomial (linear) trend.
%   8. Saves the results, the trained network and the plots in "outputs".
%
% Requirements: MATLAB R2016a or later with the Deep Learning Toolbox
%               (called the Neural Network Toolbox in older releases).
%
% Run: open this file in MATLAB and click Run, or type ExchangeRatePredictor
%      in the Command Window.

clear; clc; close all;

%% 1. Folders and settings
baseDir = fileparts(mfilename('fullpath'));
dataFile = fullfile(baseDir, 'data', 'exchange_rates.csv');
outDir = fullfile(baseDir, 'outputs');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

hiddenLayerSize = 10;          % number of hidden neurons
trainFcn = 'trainlm';          % Levenberg-Marquardt back-propagation
randomSeed = 1;                % fixed seed so that results can be repeated

%% 2. Load the data
T = readtable(dataFile);
Currencies = {'USD', 'EUR', 'GBP', 'CAD'};
Year = T.Year';                                   % 1 x N
Rates = [T.USD, T.EUR, T.GBP, T.CAD]';            % 4 x N (one column per year)
N = numel(Year);
fprintf('Loaded %d years of data (%d to %d) for %s.\n', N, Year(1), Year(end), strjoin(Currencies, ', '));

%% 3. Inputs (year t) and targets (year t+1)
x = Rates(:, 1:end-1);
t = Rates(:, 2:end);

%% 4. Create the network
rng(randomSeed);
net = fitnet(hiddenLayerSize, trainFcn);
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio   = 20/100;
net.divideParam.testRatio  = 20/100;
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 6;

%% 5. Train the network
[net, tr] = train(net, x, t);

%% 6. Evaluate the network
y = net(x);
e = gsubtract(t, y);
performance = perform(net, t, y);
fprintf('\nOverall performance (mean squared error): %.4f\n', performance);

testIdx = tr.testInd;
if isempty(testIdx)
    testIdx = 1:size(t, 2);    % fall back to all samples if no test set
end
testT = t(:, testIdx);
testY = y(:, testIdx);
RMSE = sqrt(mean((testT - testY).^2, 2));
MAPE = mean(abs((testT - testY) ./ testT), 2) * 100;

R = zeros(4, 1);
for k = 1:4
    c = corrcoef(t(k, :), y(k, :));
    R(k) = c(1, 2);
end

%% 7. Predict the next year
nextYear = Year(end) + 1;
NNPrediction = net(Rates(:, end));

%% 8. Linear trend projection (polynomial regression of degree 1)
TrendProjection = zeros(4, 1);
for k = 1:4
    p = polyfit(Year, Rates(k, :), 1);
    TrendProjection(k) = polyval(p, nextYear);
end

%% 9. Show and save the results
LastActual = Rates(:, end);
Results = table(Currencies', LastActual, NNPrediction, TrendProjection, RMSE, MAPE, R, ...
    'VariableNames', {'Currency', 'Actual_Last_Year', 'NN_Prediction', ...
                      'Linear_Trend', 'Test_RMSE', 'Test_MAPE_percent', 'R_all_data'});
fprintf('\nPredicted Naira exchange rates for %d (Naira per unit of currency):\n', nextYear);
disp(Results);

writetable(Results, fullfile(outDir, 'predictions.csv'));
save(fullfile(outDir, 'trained_network.mat'), 'net', 'tr', 'Results');

%% 10. Plots
figure; plotperform(tr);      saveas(gcf, fullfile(outDir, 'performance.png'));
figure; plottrainstate(tr);   saveas(gcf, fullfile(outDir, 'training_state.png'));
figure; ploterrhist(e);       saveas(gcf, fullfile(outDir, 'error_histogram.png'));
figure; plotregression(t, y); saveas(gcf, fullfile(outDir, 'regression.png'));

f = figure('Name', 'Exchange Rate Prediction', 'Color', 'w');
markers = {'-o', '-s', '-^', '-d'};
hold on;
for k = 1:4
    plot(Year, Rates(k, :), markers{k}, 'LineWidth', 1.2);
end
set(gca, 'ColorOrderIndex', 1);
for k = 1:4
    plot([Year(end), nextYear], [Rates(k, end), NNPrediction(k)], '--', 'LineWidth', 1.2);
end
hold off;
grid on;
xlabel('Year');
ylabel('Naira per unit of currency');
title(sprintf('Exchange Rate of Naira to USD, EUR, GBP and CAD with %d Prediction', nextYear));
legend([Currencies, strcat(Currencies, ' prediction')], 'Location', 'northwest');
saveas(f, fullfile(outDir, 'exchange_rate_prediction.png'));

fprintf('Results, trained network and plots saved in: %s\n', outDir);
