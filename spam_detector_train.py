import pandas as pd # Importing Pandas for data manipulation
import matplotlib.pyplot as plt



# Importing train_test_split for splitting the dataset into training and testing sets  
from sklearn.model_selection import train_test_split
# Importing CountVectorizer for converting text to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.naive_bayes import MultinomialNB # Importing the Multinomial Naive Bayes classifier
model = MultinomialNB() # Initializing the model

from sklearn.metrics import classification_report # shows performance metrics (precision recall f1-score support etc)
from sklearn.metrics import confusion_matrix # Gives us a matrix with counts for each true/fasle pos/neg outcome

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Importing metrics for evaluation

import pickle # This will let us store the model after training



# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Rename the columns
data.columns = ['isspam', 'message', 'unnamed1', 'unnamed2', 'unnamed3']

#print(data.head())

# Converting the 'isspam' column to binary values (1 for spam, 0 for ham)
data['isspam'] = data['isspam'].map({'ham': 0, 'spam': 1})

#print(data['isspam'].value_counts())

vectorizer = CountVectorizer(stop_words='english') # Removing stop words

# Transforming the 'message' column into a matrix of token counts
X = vectorizer.fit_transform(data['message']) 

# At this point X contains rows (messaage number, word number) and the amount of times the word appeared in the messgage.

Y = data['isspam'] # Target variable (1 for spam, 0 for ham)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # Splitting the data 80/20 with seed 42

model.fit(X_train, Y_train) # Training the model

# Now quickly saving the model using pickle so we can reuse later
with open("SpamModel.pkl", 'wb') as file:  # Creating a file with .pkl extension as write binary (w + b)
    pickle.dump(model, file) # Dumping the pickled model into the file

with open("vectoriser", 'wb') as file:  # Creating a file with .pkl extension as write binary (w + b)
    pickle.dump(vectorizer, file) # Dumping the pickled model into the file

Y_predict = model.predict(X_test) # Predicting the labels for the test set

# Now that model is trained, we evaluate the performance
Accuracy = accuracy_score(y_true=Y_test,y_pred=Y_predict)
#print(Accuracy)

Outcomes = confusion_matrix(y_true=Y_test, y_pred=Y_predict)

plt.imshow(Outcomes)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks([0,1], ["Ham", "Spam"])
plt.yticks([0,1], ["Ham", "Spam"])

plt.colorbar()
plt.show()