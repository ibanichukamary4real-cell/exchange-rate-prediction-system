function ExchangeRateApp
%EXCHANGERATEAPP Graphical user interface of the Exchange Rate Prediction System.
%
%   Lets a user with no programming knowledge load the exchange rate data,
%   train the neural network and see the predicted rates for the next year.
%
%   Run: type ExchangeRateApp in the MATLAB Command Window.
%   Requires MATLAB R2018b or later with the Deep Learning Toolbox.
%
%   Author: Lobarine Emmanuel Kinane

    baseDir = fileparts(mfilename('fullpath'));
    if ~exist(fullfile(baseDir, 'outputs'), 'dir')
        mkdir(fullfile(baseDir, 'outputs'));
    end
    currencies = {'USD', 'EUR', 'GBP', 'CAD'};
    years = [];
    rates = [];
    net = [];
    tr = [];

    % ---------------- Window and controls ----------------
    fig = uifigure('Name', 'Exchange Rate Prediction System', 'Position', [100 100 960 600]);

    uilabel(fig, 'Text', 'Exchange Rate Prediction Using Neural Network', ...
        'Position', [20 555 700 30], 'FontSize', 18, 'FontWeight', 'bold');
    uilabel(fig, 'Text', 'Naira per unit of foreign currency (USD, EUR, GBP, CAD)', ...
        'Position', [20 530 700 22]);

    uibutton(fig, 'Text', '1. Load Data', 'Position', [20 490 140 30], ...
        'ButtonPushedFcn', @(~, ~) loadData(false));
    uibutton(fig, 'Text', 'Open Other CSV...', 'Position', [170 490 140 30], ...
        'ButtonPushedFcn', @(~, ~) loadData(true));
    btnTrain = uibutton(fig, 'Text', '2. Train Network', 'Position', [320 490 140 30], ...
        'Enable', 'off', 'ButtonPushedFcn', @(~, ~) trainNetwork());
    btnPredict = uibutton(fig, 'Text', '3. Predict Next Year', 'Position', [470 490 160 30], ...
        'Enable', 'off', 'ButtonPushedFcn', @(~, ~) predictNextYear());

    uilabel(fig, 'Text', 'Hidden neurons:', 'Position', [650 490 100 30]);
    spnHidden = uispinner(fig, 'Value', 10, 'Limits', [2 50], 'Step', 1, ...
        'Position', [750 494 70 24]);

    dataTable = uitable(fig, 'Position', [20 170 430 310], ...
        'ColumnName', [{'Year'}, currencies]);

    ax = uiaxes(fig, 'Position', [470 170 470 310]);
    title(ax, 'Exchange Rate of the Naira');
    xlabel(ax, 'Year');
    ylabel(ax, 'Naira');
    grid(ax, 'on');

    resultTable = uitable(fig, 'Position', [20 50 920 110], ...
        'ColumnName', {'Currency', 'Last actual rate', 'Neural network prediction', ...
                       'Linear trend', 'Test RMSE', 'Test MAPE (%)'});

    lblStatus = uilabel(fig, 'Position', [20 12 920 28], ...
        'Text', 'Click "1. Load Data" to begin.', 'FontWeight', 'bold');

    % ---------------- Callbacks ----------------
    function loadData(chooseFile)
        file = fullfile(baseDir, 'data', 'exchange_rates.csv');
        if chooseFile
            [name, folder] = uigetfile('*.csv', 'Select exchange rate CSV file', ...
                fullfile(baseDir, 'data'));
            if isequal(name, 0)
                return;
            end
            file = fullfile(folder, name);
        end
        try
            T = readtable(file);
            years = T.Year';
            rates = [T.USD, T.EUR, T.GBP, T.CAD]';
        catch err
            uialert(fig, ['Could not read the file. It must have the columns ' ...
                'Year, USD, EUR, GBP and CAD. ' err.message], 'Data error');
            return;
        end
        if numel(years) < 6
            uialert(fig, 'At least 6 years of data are needed.', 'Data error');
            return;
        end
        dataTable.Data = [years', rates'];
        cla(ax);
        plot(ax, years, rates', '-o', 'LineWidth', 1.2);
        legend(ax, currencies, 'Location', 'northwest');
        net = [];
        resultTable.Data = {};
        btnTrain.Enable = 'on';
        btnPredict.Enable = 'off';
        lblStatus.Text = sprintf('Loaded %d years of data (%d to %d). Click "2. Train Network".', ...
            numel(years), years(1), years(end));
    end

    function trainNetwork()
        lblStatus.Text = 'Training the neural network...';
        drawnow;
        x = rates(:, 1:end-1);
        t = rates(:, 2:end);
        rng(1);
        net = fitnet(spnHidden.Value, 'trainlm');
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.60;
        net.divideParam.valRatio = 0.20;
        net.divideParam.testRatio = 0.20;
        net.trainParam.showWindow = false;
        [net, tr] = train(net, x, t);
        y = net(x);
        mse = perform(net, t, y);
        save(fullfile(baseDir, 'outputs', 'trained_network_app.mat'), 'net', 'tr');
        btnPredict.Enable = 'on';
        lblStatus.Text = sprintf(['Training finished after %d epochs (stopped: %s). ' ...
            'Overall MSE = %.2f. Click "3. Predict Next Year".'], tr.num_epochs, tr.stop, mse);
    end

    function predictNextYear()
        x = rates(:, 1:end-1);
        t = rates(:, 2:end);
        y = net(x);
        idx = tr.testInd;
        if isempty(idx)
            idx = 1:size(t, 2);
        end
        rmse = sqrt(mean((t(:, idx) - y(:, idx)).^2, 2));
        mape = mean(abs((t(:, idx) - y(:, idx)) ./ t(:, idx)), 2) * 100;
        nextYear = years(end) + 1;
        pred = net(rates(:, end));
        trend = zeros(4, 1);
        for k = 1:4
            p = polyfit(years, rates(k, :), 1);
            trend(k) = polyval(p, nextYear);
        end
        data = cell(4, 6);
        for k = 1:4
            data(k, :) = {currencies{k}, round(rates(k, end), 2), round(pred(k), 2), ...
                          round(trend(k), 2), round(rmse(k), 2), round(mape(k), 2)};
        end
        resultTable.Data = data;

        cla(ax);
        hold(ax, 'on');
        plot(ax, years, rates', '-o', 'LineWidth', 1.2);
        ax.ColorOrderIndex = 1;
        plot(ax, [years(end); nextYear], [rates(:, end)'; pred'], '--*', 'LineWidth', 1.2);
        hold(ax, 'off');
        legend(ax, [currencies, strcat(currencies, ' prediction')], 'Location', 'northwest');

        Results = cell2table(data, 'VariableNames', {'Currency', 'Actual_Last_Year', ...
            'NN_Prediction', 'Linear_Trend', 'Test_RMSE', 'Test_MAPE_percent'});
        writetable(Results, fullfile(baseDir, 'outputs', 'predictions_app.csv'));
        lblStatus.Text = sprintf(['Predicted rates for %d are shown in the table. ' ...
            'Results saved in the outputs folder.'], nextYear);
    end
end
