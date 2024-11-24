import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\ai&ml\Dataset\iris.csv')
X = df.iloc[:,:-2]
Y = df.iloc[:,-1]
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size =0.05, random_state=0)
print(X)
print("splitted",Y)