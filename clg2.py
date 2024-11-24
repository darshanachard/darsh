from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
X = pd.read_csv(r"d:\ai&ml\Dataset\breast-cancer.csv")

# Check the column names
print("Columns in the dataset:", X.columns)

# Ensure 'class' exists before popping
if 'class' in X.columns:
    y = X.pop('class')
else:
    raise KeyError("The 'class' column is not found in the dataset.")

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, cmap='binary', interpolation='None')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Display confusion matrix
confusion_matrix_df = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
print("\nConfusion matrix\n")
print(confusion_matrix_df)

# Calculate precision and recall using .loc for safer indexing
precision = confusion_matrix_df.loc[1, 1] / (confusion_matrix_df.loc[1, 1] + confusion_matrix_df.loc[0, 1])
print("\nPRECISION:\n", precision)

recall = confusion_matrix_df.loc[1, 1] / (confusion_matrix_df.loc[1, 1] + confusion_matrix_df.loc[1, 0])
print("\nRECALL:\n", recall)


# Calculate F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)
print("\nF1_SCORE:\n", f1_score)
