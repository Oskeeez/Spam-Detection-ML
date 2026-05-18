import pandas as pd # Importing Pandas for data manipulation
from sklearn.feature_extraction.text import CountVectorizer # Importing CountVectorizer for converting text to a matrix of token counts

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

print(data.head())

# Rename the columns
data.columns = ['isspam', 'message', 'unnamed1', 'unnamed2', 'unnamed3']

# Converting the 'isspam' column to binary values (1 for spam, 0 for ham)
data['isspam'] = data['isspam'].map({'ham': 0, 'spam': 1})

#print(data['isspam'].value_counts())

vectorizer = CountVectorizer(stop_words='english') # Removing stop words
X = vectorizer.fit_transform(data['message']) # Transforming the 'message' column into a matrix of token counts

print(X)



