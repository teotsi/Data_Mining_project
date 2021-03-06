import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook

from itertools import product


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


import warnings

warnings.filterwarnings('ignore')
# %matplotlib inline

DATAPATH = 'train.csv'

data = pd.read_csv(DATAPATH)
data = data[['cnt']]
print(data.head(10))
X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)


# plt.figure(figsize=(17, 8))
# plt.plot(data.head(100))
# plt.title('Bicycle count')
# plt.ylabel('Count')
# plt.xlabel('Every hour')
# plt.grid(False)
# plt.show()


def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(17, 8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')

    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)


# # Here we use moving average
# # Smooth by the previous 5 days (by week)
# plot_moving_average(data.head(100), 5)
#
# # Smooth by the previous month (30 days)
# plot_moving_average(data.head(100), 30)
#
# # Smooth by previous quarter (90 days)
# plot_moving_average(data.head(100), 90, plot_intervals=True)
# plt.show()

# SARIMA
p = 0
d = 1
q = 1
P = 0
D = 1
Q = 1
s = 5

# Create a list with all possible combinations of parameters
# parameters = product(ps, qs, Ps, Qs)
# parameters_list = list(parameters)
# len(parameters_list)


# Train many SARIMA models to find the best set of parameters
def optimize_SARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float('inf')
    print('first sarima')
    for param in tqdm_notebook(parameters_list):
        try:
            model = sm.tsa.statespace.SARIMAX(X_train.head(100), order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue

        aic = model.aic

        # Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table


print('second sarima')
# result_table = optimize_SARIMA(parameters_list, d, D, s)

# Set parameters that give the lowest AIC (Akaike Information Criteria)
# p, q, P, Q = result_table.parameters[0]
print('third sarima')
best_model = sm.tsa.statespace.SARIMAX(X_train, order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)
x_pred = best_model.predict()
x_pred[x_pred<0]=0
#print(x_pred)

print(best_model.summary())
mod_Test=X_test
mod_Pred=x_pred
if len(X_test)!=len(x_pred):
    if len(X_test)>len(x_pred):
        mod_Test=X_test.head(len(x_pred))
    else:
        mod_Pred=x_pred.head(len(X_test))
print('RMSLE:', np.sqrt(mean_squared_log_error(mod_Test, mod_Pred)))
print('R2:', r2_score(mod_Test, mod_Pred))

