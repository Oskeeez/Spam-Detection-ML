import pandas as pd

data = pd.read_csv('spam.csv', encoding='latin-1')

print(data.head())

data.columns = [['isspam', 'message', 'unnamed1', 'unnamed2', 'unnamed3']]

data['isspam'] = data['isspam'].map(lambda x: 1 if x == 'spam' else 0)

print(data['isspam'].value_counts())



