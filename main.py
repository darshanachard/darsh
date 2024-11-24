import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics

# Load the dataset
df = pd.read_csv(r'd:\ai&ml\Dataset\iris.csv')
print(df.columns)

# Visualizing sepal dimensions
plt.xlabel('sepal length')
plt.ylabel('sepal width (cm)')
plt.scatter(df['sepal.length'], df['sepal.width'], color='green')
plt.show()

# Visualizing petal dimensions
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df['petal.length'], df['petal.width'], color='red')
plt.scatter(df['petal.length'], df['petal.width'], color='blue')
plt.show()

# Preparing data for logistic regression
x = df.drop(['variety'], axis='columns')  # Corrected column name
y = df['variety']  # Corrected column name
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training the model
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

# Evaluating the model
score = model.score(x_test, y_test)
print(f'Model Score: {score}')
