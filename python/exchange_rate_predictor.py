"""
exchange_rate_predictor.py - free Python version of the Exchange Rate Prediction System.

Use this if MATLAB is not available. It follows the same method as
ExchangeRatePredictor.m: a multilayer perceptron with one hidden layer of 10
tanh neurons learns to predict next year's Naira rates (USD, EUR, GBP, CAD)
from this year's rates, and a linear trend is fitted for comparison.

Run:  pip install -r requirements.txt
      python exchange_rate_predictor.py
Results and plots are saved in the outputs folder of the project.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "exchange_rates.csv")
OUT = os.path.join(BASE, "outputs")
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]
SEED = 1


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    years = df["Year"].to_numpy()
    rates = df[CURRENCIES].to_numpy(dtype=float)          # N x 4
    print("Loaded %d years of data (%d to %d)." % (len(years), years[0], years[-1]))

    X, Y = rates[:-1], rates[1:]                           # year t -> year t+1
    sx = MinMaxScaler((-1, 1)).fit(X)
    sy = MinMaxScaler((-1, 1)).fit(Y)
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

    model = MLPRegressor(hidden_layer_sizes=(10,), activation="tanh", solver="lbfgs",
                         alpha=1e-3, max_iter=2000, random_state=SEED)
    model.fit(sx.transform(X[tr_idx]), sy.transform(Y[tr_idx]))

    pred_all = sy.inverse_transform(model.predict(sx.transform(X)))
    err = Y[te_idx] - pred_all[te_idx]
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mape = (np.abs(err / Y[te_idx])).mean(axis=0) * 100

    next_year = int(years[-1]) + 1
    nn_next = sy.inverse_transform(model.predict(sx.transform(rates[-1:])))[0]
    trend = [np.polyval(np.polyfit(years, rates[:, k], 1), next_year) for k in range(4)]

    res = pd.DataFrame({
        "Currency": CURRENCIES,
        "Actual_Last_Year": rates[-1].round(2),
        "NN_Prediction": np.round(nn_next, 2),
        "Linear_Trend": np.round(trend, 2),
        "Test_RMSE": rmse.round(2),
        "Test_MAPE_percent": mape.round(2),
    })
    print("\nPredicted Naira exchange rates for %d:" % next_year)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUT, "predictions_python.csv"), index=False)

    plt.figure(figsize=(8, 5))
    styles = ["-o", "-s", "-^", "-d"]
    for k, c in enumerate(CURRENCIES):
        line, = plt.plot(years, rates[:, k], styles[k], label=c)
        plt.plot([years[-1], next_year], [rates[-1, k], nn_next[k]], "--", color=line.get_color(),
                 label=c + " prediction")
    plt.grid(True, ls=":")
    plt.xlabel("Year")
    plt.ylabel("Naira per unit of currency")
    plt.title("Exchange Rate of Naira to USD, EUR, GBP and CAD with %d Prediction" % next_year)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exchange_rate_prediction_python.png"), dpi=150)
    print("\nResults saved in", OUT)


if __name__ == "__main__":
    main()
