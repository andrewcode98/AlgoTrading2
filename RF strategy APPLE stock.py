# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:56:57 2024

@author: andre
"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import empyrical as ep
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import cross_val_score, KFold
from arch import arch_model
from sklearn.metrics import accuracy_score
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

start_sp = datetime.datetime(2012, 12, 10)
end_sp = datetime.datetime.today()
APPLE = yf.download('AAPL', start_sp, end_sp)
X_path = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Algorithmic Trading\EFFR.csv'
EFFR = pd.read_csv(X_path, header=0)
EFFR = EFFR.set_index('Effective Date') 
EFFR = EFFR.dropna()

# Converting index to datetime object
EFFR.index = pd.to_datetime(EFFR.index)

# Sorting by index (oldest to newest date)
EFFR = EFFR.sort_index()
# Prepare the feature data
volume = APPLE['Volume']
close = APPLE['Close']
low = APPLE['Low']
high = APPLE['High']
_open = APPLE['Open']
# Convert annual to day rate
EFFR = 1/252 * EFFR
EFFR = EFFR['Rate (%)']



starting_date = "2014-01-02"
ending_date = end_sp.strftime('%Y-%m-%d')

APPLE_close = APPLE['Close']
APPLE_returns = APPLE_close / APPLE_close.shift(1) - 1
APPLE_returns.drop(APPLE_returns.index[0], inplace=True)


# Get the indices of EFFR and APPLE_returns
effr_indices = set(EFFR.index)
APPLE_indices = set(APPLE_returns.index)

# Find indices that are in EFFR but not in APPLE_returns
indices_only_in_effr = effr_indices - APPLE_indices

# Find indices that are in APPLE_returns but not in EFFR
indices_only_in_APPLE = APPLE_indices - effr_indices


# Append rows with indices only in APPLE_returns to EFFR with NaN values
for index in indices_only_in_APPLE:
    EFFR.loc[index] = np.nan

# Sorting by index (oldest to newest date)
EFFR = EFFR.sort_index()
    
# Replace NaN values in EFFR with previous index values
EFFR.fillna(method='ffill', inplace=True)

## Remove indices_only_in_effr from EFFR
EFFR.drop(indices_only_in_effr, inplace=True)

# Perform subtraction
APPLE_excess_returns = APPLE_returns - EFFR
APPLE_excess_returns = APPLE_excess_returns.dropna()
APPLE_excess_returns = APPLE_excess_returns[(APPLE_excess_returns.index >= starting_date) & (APPLE_excess_returns.index <= ending_date)]


def plot_time_series(time_series,save_path):
    
 
    
    
    plt.figure(figsize=(10, 6))


    
    plt.plot(time_series.index, time_series)
    plt.tight_layout()
    plt.title('APPLE daily returns', fontsize = 18)
    plt.ylabel("APPLE return", fontsize = 16)
    plt.xlabel("Date", fontsize = 16)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if save_path:
        plt.savefig(save_path, format='pdf')
    plt.show()    
    
save_path_1 = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\APPLE_returns.pdf'
save_path_2 = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\EFFT rate.pdf'
save_path_3 = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\APPLE_excess_return.pdf '  

# Plot the data
plot_time_series(APPLE_returns, save_path_1)

plt.figure(figsize=(10, 6))




plt.plot(APPLE_close)


# On balance volume indicator
OBV = pd.Series(0, index=volume.index)


for i,date in enumerate(volume.index):
    if i == 0:
        continue
    if close.iloc[i] < close.iloc[i-1]:
        OBV.iloc[i] = OBV.iloc[i-1] - volume.iloc[i]
    elif close.iloc[i] > close.iloc[i-1]:
        OBV.iloc[i] = OBV.iloc[i-1] + volume.iloc[i]
    else:
        OBV.iloc[i] = OBV.iloc[i-1]
      
lowest_low = low.rolling(window=14).min()
highest_high = high.rolling(window=14).max()

# Calculate %K
percent_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100

# Create a Series for Stochastic Oscillator
SO = pd.Series(percent_k, index=close.index)

# MACD
ema_12 = close.ewm(span=12, adjust=False).mean()
ema_26 = close.ewm (span=26, adjust=False).mean()
MACD = ema_12 - ema_26

def compute_moving_averages(window,time_series):
    return time_series.rolling(window=window).mean()
      
MA_20 = compute_moving_averages(20, APPLE_close)
MA_20 = MA_20.rename("MA_20")

MA_5 = compute_moving_averages(5, APPLE_close)
MA_5 = MA_20.rename("MA_5")

SO = SO.rename("SO")


OBV = OBV.pct_change().dropna()
OBV = OBV.rename("OBV")

MACD = MACD.rename("MACD")

MA_20 = MA_20[(MA_20.index >= starting_date) & (MA_20.index <= ending_date)]
MA_5 = MA_5[(MA_5.index >= starting_date) & (MA_5.index <= ending_date)]
OBV = OBV[(OBV.index >= starting_date) & (OBV.index <= ending_date)]
MACD = MACD[(MACD.index >= starting_date) & (MACD.index <= ending_date)]
SO = SO[(SO.index >= starting_date) & (SO.index <= ending_date)]

corr_features = pd.concat([SO, OBV, MACD, MA_20, MA_5], axis=1)
# Compute the correlation matrix
correlation_matrix = corr_features.corr()

print(correlation_matrix)

plt.figure()
OBV.plot()
SO.plot()
MACD.plot()
plt.show()

def augmented_dickey_fuller_test(time_series,alpha,indicator):
    
    ad_fuller_stat = adfuller(time_series)[0]
    p_value = adfuller(time_series)[1]
    if p_value <= alpha:
        print(f"{indicator}: p_value: {p_value}, Critical: {ad_fuller_stat}. Time series is stationary")
    else:
        print(f"{indicator} p_value: {p_value}, Critical: {ad_fuller_stat}. Time series is not stationary")
        
        

# Resample 'close' to get return after 6 trading days
close = close[close.index>=starting_date]

# Calculate returns
returns = close / close.shift(-1) - 1


    




returns = returns[(returns.index >= '2014-01-02') & (returns.index <= ending_date)]


returns = returns.rename('Movement')
# Start SO, OBV, and MACD from index 14 to remove nan entries






# Change stock movement to +1, -1 if stock goes up or down
for i,date in enumerate(returns.index):
    if returns.iloc[i] < 0:
        returns.iloc[i] = 0
    else:
        returns.iloc[i] = 1
        
        
#Merge DataFrames
merged_df = pd.concat([SO, OBV, MACD, MA_20, MA_5], axis=1)

merged_df.index = returns.index

# Split the data into training and testing sets
# Define the split ratio
test_size = 0.3
train_size = 1 - test_size

# Calculate the number of training samples
n_train = int(len(merged_df) * train_size)

# Manually split the data
X_train = merged_df.iloc[:n_train]
X_test = merged_df.iloc[n_train:]
y_train = returns.iloc[:n_train]
y_test = returns.iloc[n_train:]




# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=43, max_features=3, random_state=42, max_depth=30)


# Fit the model on the training data
rf_clf.fit(X_train, y_train.values.ravel())

# Compute the permutation feature importance
result_2 = permutation_importance(rf_clf, X_train, y_train, scoring='accuracy', n_repeats=100, random_state=42, n_jobs=-1)

# Sort the features by importance
sorted_idx = result_2.importances_mean.argsort()

movement_pred = rf_clf.predict(merged_df)
movement_pred_train = rf_clf.predict(X_train)
movement_pred_test = rf_clf.predict(X_test)

accuracy_train = accuracy_score(y_train,movement_pred_train)
print ("Accuracy in training set: ", accuracy_train)
accuracy_test = accuracy_score(y_test,movement_pred_test)
print("Accuracy in test set: " , accuracy_test)

# Plot the feature importance
# plt.figure(figsize=(12, 8))
# plt.boxplot(result_2.importances[sorted_idx].T, vert=False, labels=X_train.columns[sorted_idx])
# plt.title("Permutation Feature Importance (Random Forest)", fontsize = 18)
# plt.xlabel("Feature Importance Score in Training Set" , fontsize = 16)
# plt.tight_layout()
# plt.tick_params(axis='x', labelsize=14)
# plt.tick_params(axis='y', labelsize=14)
# plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Machine Learning\Permutation_Score.pdf', format='pdf')
# plt.show()


n_estimators_range = range(21,100)
# Lists to store scores
val_accuracies_estimators = []
val_f1_scores_estimators = []

# Manually split the training data further into training and validation
n_train = int(len(X_train) * train_size)
X_train_samp = X_train.iloc[:n_train]
X_val_samp = X_train.iloc[n_train:]
y_train_samp = y_train.iloc[:n_train]
y_val_samp = y_train.iloc[n_train:]
# oob_scores_estimators = []
# for n_estimators in n_estimators_range:
#     rf_clf = RandomForestClassifier(n_estimators=n_estimators, 
#                                     oob_score=True, 
#                                     max_features=3, 
#                                     random_state=42,
#                                     n_jobs=-1,
#                                     warm_start=True) # Enable warm_start to add more trees incrementally
    
#     # Train the model on the training split
#     rf_clf.fit(X_train_samp, y_train_samp.values.ravel())
    
#     # Calculate accuracy on the validation set
#     val_accuracy = rf_clf.score(X_val_samp, y_val_samp)
#     val_accuracies_estimators.append(val_accuracy)
    
    
#     # Calculate and append the OOB score
#     oob_scores_estimators.append(rf_clf.oob_score_)

# # Find the index of the highest scores

# best_accuracy_index = np.argmax(val_accuracies_estimators)
# best_oob_index = np.argmax(oob_scores_estimators)

# # Plotting the metrics
# plt.figure(figsize=(12, 8))

# # Plot OOB score
# plt.plot(n_estimators_range, oob_scores_estimators, '-o', label='OOB Score')
# plt.plot(n_estimators_range[best_oob_index], oob_scores_estimators[best_oob_index], 'ro')  # Highlight highest OOB score

# # Plot validation accuracy
# plt.plot(n_estimators_range, val_accuracies_estimators, '-o', label='Validation Accuracy')
# plt.plot(n_estimators_range[best_accuracy_index], val_accuracies_estimators[best_accuracy_index], 'ro')  # Highlight highest accuracy


# plt.xlabel('Number of Trees (n_estimators)', fontsize = 16)
# plt.ylabel('Metrics', fontsize = 16)
# plt.title('Performance Metrics Across Different Numbers of Trees', fontsize = 18)

# # Print the highest scores with their corresponding n_estimators
# print(f'Highest OOB Score: {oob_scores_estimators[best_oob_index]:.3f} (n_estimators={n_estimators_range[best_oob_index]})')
# print(f'Highest Validation Accuracy: {val_accuracies_estimators[best_accuracy_index]:.3f} (n_estimators={n_estimators_range[best_accuracy_index]})')


# plt.legend(fontsize = 14)
# plt.grid(True)
# plt.tick_params(axis='x', labelsize=14)
# plt.tick_params(axis='y', labelsize=14)
# plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Machine Learning\N_estimators.pdf', format='pdf')
# plt.show()

# Lists to store scores

# Lists to store scores
# val_accuracies_depth = []
# val_oob_score_depth = []

# max_depth_range = range(1, 40)  # Change the range as needed

# for max_depth in max_depth_range:
#     rf_clf = RandomForestClassifier(n_estimators=43, 
#                                     oob_score=True, 
#                                     max_features=3, 
#                                     random_state=42,
#                                     n_jobs=-1,
#                                     warm_start=True,
#                                     max_depth=max_depth)
    
#     # Train the model on the resampled data
#     rf_clf.fit(X_train_samp, y_train_samp.values.ravel())
    
#     # Calculate accuracy on the validation set
#     val_accuracy = rf_clf.score(X_val_samp, y_val_samp)
#     val_accuracies_depth.append(val_accuracy)
#     # Calculate and append the OOB score
#     val_oob_score_depth.append(rf_clf.oob_score_)

# # Plotting the metrics
# plt.figure(figsize=(12, 8))

# # Plot validation accuracy
# plt.plot(max_depth_range, val_accuracies_depth, '-o', label='Validation Accuracy')
# plt.plot(max_depth_range, val_oob_score_depth, '-o', label='OOB Accuracy')



# plt.xlabel('Max Depth', fontsize = 16)
# plt.ylabel('Metrics', fontsize = 16)
# plt.title('Performance Metrics Across Different Max Depths', fontsize = 18)
# plt.legend(fontsize = 14)
# plt.grid(True)
# plt.tick_params(axis='x', labelsize=14)
# plt.tick_params(axis='y', labelsize=14)
# plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Machine Learning\Max_Depth.pdf', format='pdf')
# plt.show()



# Plot ACF and PACF of absolute returns of APPLE
returns_squared = APPLE_returns ** 2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'hspace': 0.5})
plot_acf(returns_squared, ax=ax1, lags=10)
ax1.set_ylim(-.5, .5) 
ax1.set_title(r"Autocorrelation in $r_{t}^2$", fontsize=18)
ax1.set_xlabel("Lag", fontsize=16)  # Set x-axis label for ax1
ax1.set_xticks(range(0, 11))
ax1.set_xticklabels(range(0, 11))  # Set x-axis tick labels for ax1
plot_pacf(returns_squared, ax=ax2, lags=10)
ax2.set_ylim(-.5, .5)  
ax2.set_title(r"Partial Autocorrelation in $r_{t}^2$", fontsize=18)
ax2.set_xlabel("Lag", fontsize=16)  # Set x-axis label for ax2
ax2.set_xticks(range(0, 11))
ax2.set_xticklabels(range(0, 11))  # Set x-axis tick labels for ax2



APPLE_returns = APPLE_returns[APPLE_returns.index >= starting_date ]
# Split the data into training and testing sets
# Define the split ratio
test_size = 0.3
train_size = 1 - test_size
val_size = 0.2
# Calculate the number of training samples
n_train = int(len(APPLE_returns) * train_size)
n_val = int(n_train * val_size)
# Manually split the data
x_train = APPLE_returns.iloc[:n_train].values
x_train_series = APPLE_returns.iloc[:n_train]
x_test_series = APPLE_returns.iloc[n_train:]
x_test = APPLE_returns.iloc[n_train:].values
x_val = APPLE_returns.iloc[:n_val].values
x_train = APPLE_returns.iloc[n_val:].values



plt.figure()
plt.hist(APPLE_returns, bins = 100)
plt.show()
# Looks like a student - t distribution

fig, ax = plt.subplots()
stats.probplot(APPLE_returns, dist="t", sparams=(4,), plot=ax)
ax.set_title('Q-Q Plot for daily APPLE returns')
plt.show()

# Fit a GARCH(1,1) model with Student's t-distribution 
model = arch_model(x_train_series*100, vol='Garch', p=1, q=1, dist='t')
model_fit = model.fit(disp='off')

rolling_preds = []
# Predict conditional volatility 
for i in range(x_test.shape[0]):
    train = 100 * APPLE_returns[:-(x_test.shape[0]-i)]
    model = arch_model(train, p=1, q=1,  vol='GARCH', dist='t')
    model_fit = model.fit(disp='off')
    # One step ahead predictor
    pred = model_fit.forecast(horizon=1, reindex=True)
    rolling_preds.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_preds = pd.Series(rolling_preds, index=x_test_series.index)



# Print the model summary
# print(model_fit.summary())

# Plot the conditional volatility

plt.plot(model_fit.conditional_volatility/50, color = 'black', alpha = 0.5, label = 'Scaled Volatility train')
plt.plot(rolling_preds/50 , color = 'blue', label = 'Scaled Volatility test')
plt.plot(abs(APPLE_returns), color = 'red', alpha = 0.5, label = 'Absolute APPLE returns ')
plt.title('Conditional Volatility')
plt.legend()
plt.show()

training_volatility = model_fit.conditional_volatility
test_volatility = rolling_preds
conditional_volatility = pd.concat([training_volatility,test_volatility])


def ml_strategy_positions(movement):
    positions = []
    movement = np.array(movement)
    for i in range(len(movement)):
        if movement[i] == 1:  # Buy if forecast upward movement for the next horizon
            positions.append(1)  
            
            
        elif movement[i] == 0:  # Sell if forecast downward movement for the next horizon
            positions.append(-1)  
          
            
    return positions




positions = ml_strategy_positions(movement_pred)
# To match the indices
# for i in range(abs(len(positions) - len(APPLE_excess_returns))):
#     positions.append(0)


inverse_vol = 1/conditional_volatility
# Min-Max scaling
min_val = np.min(inverse_vol)
max_val = np.max(inverse_vol)
scaled_position = (inverse_vol - min_val) / (max_val - min_val) * 1/10

positions = pd.Series(positions, index = APPLE_excess_returns.index)
scaled_position = scaled_position[~scaled_position.index.duplicated()]
scaled_position = scaled_position[~scaled_position.index.duplicated()]
strategy_dataframe = pd.DataFrame({"Buy/Sell":positions, "Fraction":scaled_position})


# Merge positions and close prices into a single DataFrame
df = pd.DataFrame({'Close': APPLE_close, 'Positions': positions})

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], color='black', lw=2, label='Close Price')

# Plotting buy positions (green triangles)
buy_positions = df[df['Positions'] == 1]
plt.scatter(buy_positions.index, buy_positions['Close'], color='green', marker='^', label='Buy Position')

# Plotting sell positions (red triangles)
sell_positions = df[df['Positions'] == -1]
plt.scatter(sell_positions.index, sell_positions['Close'], color='red', marker='v', label='Sell Position')

plt.title('Machine Learning Strategy Positions vs Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

def ml_strategy_theta(positions, initial_capital, margin, excess_returns, sizing):
    excess_returns = np.array(excess_returns)
    portfolio_excess_returns = []
    V0 = 200000
    portfolio_value = [V0]
    positions = np.array(positions)
    theta = []
    for i in range(len(positions)):
        if not theta:
            theta.append((positions[i] * portfolio_value[-1] * 1/10))
            # Max position in theta is 2 million in both long and short side
        elif positions[i] == 1:
            if abs(portfolio_value[-1] * sizing.iloc[i]) > margin:
                theta.append(theta[-1])
            else:
                theta.append( + (portfolio_value[-1] * sizing.iloc[i]) ) # buy position
        elif positions[i] == -1:
            if abs(portfolio_value[-1] * sizing.iloc[i]) > margin:
                theta.append(theta[-1])
            else:
                theta.append( - (portfolio_value[-1] * sizing.iloc[i]) )# sell position
        else:
            theta.append(theta[-1])# hold position
            
        portfolio_value.append(portfolio_value[-1] + (theta[i] * excess_returns[i]))
        portfolio_excess_returns.append((theta[i] * excess_returns[i])/portfolio_value[-1])
    return theta


def daily_trading_pnl(theta,excess_daily_returns):
    returns = np.array(excess_daily_returns)
    theta = np.array(theta)
    daily_pnl = []
    for i in range(len(theta)):
        daily_pnl.append(theta[i] * returns[i])
    return daily_pnl

initial_capital = 200000
L = 10 * initial_capital
date = X_train.index[-1].strftime('%Y-%m-%d')
theta = ml_strategy_theta(positions, initial_capital , L, APPLE_excess_returns, scaled_position)
theta = pd.Series(theta, index=APPLE_excess_returns.index)


theta_train = theta[theta.index<=date]
theta_test = theta[theta.index>date]
APPLE_excess_returns_train = APPLE_excess_returns[APPLE_excess_returns.index <= date] 
APPLE_excess_returns_test =  APPLE_excess_returns[APPLE_excess_returns.index > date] 
daily_pnl_train = daily_trading_pnl(theta_train,APPLE_excess_returns_train)
daily_pnl_train = pd.Series(daily_pnl_train, index=APPLE_excess_returns_train.index)
daily_pnl_test = daily_trading_pnl(theta_test,APPLE_excess_returns_test)
daily_pnl_test = pd.Series(daily_pnl_test, index=APPLE_excess_returns_test.index)
cumulated_pnl_dv = pd.concat([daily_pnl_train,daily_pnl_test])
cumulated_pnl_dv = cumulated_pnl_dv.cumsum()
cumulated_pnl_dv_train = cumulated_pnl_dv[cumulated_pnl_dv.index <= date ]
cumulated_pnl_dv_test = cumulated_pnl_dv[cumulated_pnl_dv.index > date ]

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot for Daily Excess Profit and Loss
axs[0].plot(APPLE_excess_returns_train.index, daily_pnl_train, color='blue', label='In-Sample')
axs[0].plot(APPLE_excess_returns_test.index, daily_pnl_test, color='green', label='Out-of-Sample')
axs[0].set_title('Daily Excess Profit and Loss', fontsize=18)
axs[0].set_ylabel("Profit and Loss", fontsize=16)
axs[0].set_xlabel("Date", fontsize=16)
axs[0].grid()
axs[0].legend(fontsize=16)
axs[0].tick_params(labelsize=14)

# Plot for Cumulative Excess Profit and Loss
axs[1].plot(APPLE_excess_returns_train.index, cumulated_pnl_dv_train/10**6, color='blue', label='In-Sample')
axs[1].plot(APPLE_excess_returns_test.index, cumulated_pnl_dv_test/10**6, color='green', label='Out-of-Sample')
axs[1].set_title('Cumulative Excess Profit and Loss', fontsize=18)
axs[1].set_ylabel("Cumulative Excess Profit and Loss (millions)", fontsize=16)
axs[1].set_xlabel("Date", fontsize=16)
axs[1].grid()
axs[1].legend(fontsize=16)
axs[1].tick_params(labelsize=14)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\ML_Excess_PNL.pdf', format='pdf')
plt.tight_layout()
plt.show()


# Plot theta positions with bounds:
def theta_positions(theta_train,theta_test,V0,L):
    plt.figure(figsize=(10, 6))
    upper_bound = V0 * L
    lower_bound = -V0 * L
    theta_combined = pd.concat([theta_train, theta_test])
    plt.plot(theta_train.index, theta_train/10**6, color='blue', label = 'In-Sample')
    plt.plot(theta_test.index, theta_test/10**6, color='green', label = 'Out-of-Sample')
    plt.plot(theta_combined.index, np.ones_like(theta_combined) * upper_bound /10**6, color='red', linestyle='--', label='Upper Bound')
    plt.plot(theta_combined.index, np.ones_like(theta_combined) * lower_bound /10**6, color='red', linestyle='--', label='Lower Bound')
    plt.tight_layout()
    plt.title('$\\theta$ positions in dollars against time', fontsize = 18)
    plt.ylabel("$\\theta$ (millions)", fontsize=16)
    plt.xlabel("Date", fontsize = 16)
    plt.grid()
    plt.legend(loc='upper right')
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\ML_Theta.pdf', format='pdf')
    plt.show()      

V0 = 200000
L = 10
theta_positions(theta_train,theta_test,V0,L)

APPLE_close = APPLE_close [(APPLE_close.index >= starting_date) & (APPLE_close.index <= ending_date)]

# Calculate turnover in dollars and units
def turnover(theta,close):
    theta_difference = theta.shift(1)-theta
    theta_difference.drop(theta_difference.index[0], inplace=True)
    theta_difference = np.array(theta_difference)
    turnover_dollars = np.sum(np.abs(theta_difference))
    units = theta/close
    units_difference = units.shift(1)-units
    units_difference.drop(units_difference.index[0], inplace=True)
    units_difference = np.array(units_difference)
    turnover_units = np.sum(np.abs(units_difference))
    return turnover_dollars,turnover_units
    


turnover_dollars,turnover_units = turnover(theta,APPLE_close)
print("Turnover_dollars: ", turnover_dollars)
print("Turnover_units: ", turnover_units)

def moving_average_turnover(theta,lag):
    theta_difference = theta.shift(1)-theta
    theta_difference.drop(theta_difference.index[0], inplace=True)
    return np.abs(theta_difference).rolling(window=lag).mean()

# 30 -day Volatility of APPE

lag = 30
rolling_vol = APPLE_returns.rolling(window=lag).std()
moving_average_turnover = moving_average_turnover(theta,lag)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(moving_average_turnover.index, moving_average_turnover, color='black')
ax1.set_title('30-day Moving Average Turnover', fontsize=18)
ax1.set_ylabel("Moving Average Turnover", fontsize=16)
ax1.grid()
ax1.legend(loc='upper right')
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.plot(rolling_vol.index, rolling_vol, color='red')
ax2.set_title('30-day realized Volatility', fontsize=18)
ax2.set_xlabel("Date", fontsize=16)
ax2.set_ylabel("Volatility", fontsize=16)
ax2.grid()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\ML_Moving_Turnover.pdf', format='pdf')
plt.tight_layout()
plt.show()


daily_pnl = daily_trading_pnl(theta, APPLE_excess_returns)


portfolio_value = np.zeros(len(theta)+1)
portfolio_value[0] = V0
for i in range(len(theta)):
    portfolio_value[i+1] = portfolio_value[i] + daily_pnl[i]
# Sharpe ratio
portfolio_excess_returns =  (portfolio_value[1:] - portfolio_value[:-1]) / np.abs(portfolio_value[:-1])
portfolio_excess_returns = pd.Series(portfolio_excess_returns, index = APPLE_excess_returns.index)


port_excess_returns_train = portfolio_excess_returns[portfolio_excess_returns.index <= '2021-04-09']
port_excess_returns_test = portfolio_excess_returns[portfolio_excess_returns.index > '2021-04-09']
sharpe_train = ep.sharpe_ratio(port_excess_returns_train, period='daily')
sharpe_test = ep.sharpe_ratio(port_excess_returns_test)
sortino_train = ep.sortino_ratio(port_excess_returns_train)
sortino_test = ep.sortino_ratio(port_excess_returns_test)
maximum_drawdown_train = ep.max_drawdown(port_excess_returns_train)
maximum_drawdown_test = ep.max_drawdown(port_excess_returns_test)
calmar_train = ep.calmar_ratio(port_excess_returns_train)
calmar_test = ep.calmar_ratio(port_excess_returns_test)

print(f"Sharpe Ratio (In-sample): {sharpe_train:.2f}")
print(f"Sharpe Ratio (Out-of-sample):  {sharpe_test:.2f}")
print(f"Sortino Ratio (In-sample): {sortino_train:.2f}")
print(f"Sortino Ratio (Out-of-sample): {sortino_test:.2f}")
print(f"Max Drawdown (In-sample): {maximum_drawdown_train:.3f}")
print(f"Max Drawdown (Out-of-sample): {maximum_drawdown_test:.3f}")
print(f"Calmar Ratio (In-sample): {calmar_train:.2f}")
print(f"Calmar Ratio (Out-of-sample): {calmar_test:.2f}")

window = 60
rolling_mean = portfolio_excess_returns.rolling(window=window).mean()
rolling_std = portfolio_excess_returns.rolling(window=window).std()
rolling_sharpe = (rolling_mean / rolling_std) 
rolling_sharpe.fillna(0, inplace=True) # To deal with discontinuities when std is zero
rolling_sharpe_train = rolling_sharpe[rolling_sharpe.index <= '2021-04-09']
rolling_sharpe_test = rolling_sharpe[rolling_sharpe.index > '2021-04-09']

plt.figure(figsize=(10, 6))
plt.plot(rolling_sharpe_train.index, rolling_sharpe_train, color='blue', label = 'In-Sample')
plt.plot(rolling_sharpe_test.index, rolling_sharpe_test, color='green', label = 'Out-of-Sample')
plt.tight_layout()
plt.title('Rolling-60-day Sharpe Ratio', fontsize = 18)
plt.ylabel("Sharpe Ratio", fontsize = 16)
plt.xlabel("Date", fontsize = 16)
plt.grid()
plt.legend()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\ML_Rolling_Sharpe.pdf', format='pdf')
plt.show()   


# Plotting the return distribution
plt.figure(figsize=(10, 6))
plt.hist(daily_pnl, bins=30, edgecolor='black', alpha=0.7)
plt.title('ML Return Distribution')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

def drawdown(daily_pnl):
    daily_pnl = np.array(daily_pnl)
    drawdown_array = np.zeros(len(daily_pnl))
    for i in range(len(daily_pnl)):
        drawdown_array[i] = np.max(daily_pnl[:i+1]) - daily_pnl[i]
    return drawdown_array

drawdown_t = drawdown(daily_pnl)
drawdown_t = pd.Series(drawdown_t, index=APPLE_excess_returns.index)    
lag = 90
historical_90_vol = APPLE_returns.rolling(window=lag).std()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(drawdown_t.index, drawdown_t, color='black')
ax1.set_title('Drawdown chart', fontsize=18)
ax1.set_ylabel("Drawdown", fontsize=16)
ax1.grid()
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.plot(historical_90_vol.index, historical_90_vol, color='red')
ax2.set_title('90-day realized Volatility', fontsize=18)
ax2.set_xlabel("Date", fontsize=16)
ax2.set_ylabel("Volatility", fontsize=16)
ax2.grid()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\ML_Drawdown_Chart.pdf', format='pdf')
plt.tight_layout()
plt.show()

