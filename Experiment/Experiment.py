import pandas as pd 
import numpy as np 
import Reader 

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 13000)

df = Reader.read_data('train.csv', one_hot=False)
dfa = df.sort_values(by=['month', 'weekday'])

extra = pd.read_csv('Experiment/extra.csv')

print(df.shape[0])
cnt_avg_perMnth_perDay = [] 

for i in range(df.shape[0]):
        mnth = df.month[i]
        day = df.weekday[i]
        cnt_avg_perMnth_perDay.append(extra.iloc[mnth-1][day])

print(len(cnt_avg_perMnth_perDay))
df['Month_day_cnt_avg'] = cnt_avg_perMnth_perDay
print(df)