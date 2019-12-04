import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)


filename = 'train.csv'
df_train = pd.read_csv(filename)

print(df_train.head(10))
# Training
all_columns = list(df_train.columns)
# Training and test data is created by splitting the main data. 30% of test data is considered
train_columns = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'humidity']
X = df_train[[x for x in all_columns if x.startswith(tuple(train_columns))]]  # getting all desired
# print(X)

plt.figure(figsize=(17, 8))
plt.plot(df_train.head(100))
plt.title('Bicycles')
plt.ylabel('Count')
plt.xlabel('Hours')
plt.grid(False)
plt.show()
