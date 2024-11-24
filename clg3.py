import pandas as pd
df=pd.read_csv(r'd:\ai&ml\Dataset\iris.csv')
print(df.columns)
from matplotlib import pyplot as plt
plt.xlabel('sepal length ')
plt.ylabel('sepal width (cm)')
plt.scatter(df['sepal.length'],df['sepal.width'],color='green')
plt.scatter(df['sepal.length'],df['sepal.width'],color='red',)
plt.show()
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df['petal.length'],df['petal.width'],color='red')
plt.scatter(df['petal.length'],df['petal.width'],color='blue')
plt.show()
from sklearn.model_selection import train_test_split
x = df.drop(['variety'], axis='columns')  # Adjusted to match the actual column name
y = df['variety']  # Adjusted to match the actual column name
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
len(x_train)
len(x_test)
from sklearn.svm import SVC
model = SVC()
model.fit(x, y)
model.score(x_test,y_test)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
# true labels
y_true = [0, 0, 0, 1, 1, 1]
# predicted labels
y_pred = [0, 0, 1, 1, 1, 1]
# compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
# plot confusion matrix
plt.imshow(conf_matrix, cmap='binary', interpolation='None')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
print("\nConfusion matrix\n")
print("\n")
print(confusion_matrix)
print("\n")
precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
print("\n")
print("\nPRECISION:\n")
print(precision)
print("\n")
recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print("\nRECALL:\n")
print(recall)
recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
print("\n")
print("\nF1_SCORE:\n")
f1_score = 2 * (precision * recall) / (precision + recall)
print(f1_score)