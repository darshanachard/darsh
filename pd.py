import pandas as pd 
df=pd.read_csv(r'd:\ai&ml\Dataset\iris.csv') 
df 
from matplotlib import pyplot as plt 
plt.xlabel('sepal length ') 
plt.ylabel('sepal width (cm)') 
plt.scatter(df['sepal_length'],df['sepal_width'],color='green') 
plt.scatter(df['sepal_length'],df['sepal_width'],color='red',)
plt.xlabel('petal length (cm)') 
plt.ylabel('petal width (cm)') 
plt.scatter(df['petal_length'],df['petal_width'],color='red') 
plt.scatter(df['petal_length'],df['petal_width'],color='blue') 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model, metrics 
x=df.drop(['species'],axis='columns') 
y=df.species 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1) 
model = linear_model.LogisticRegression() 
model.fit(x, y) 
model.score(x_test,y_test)
