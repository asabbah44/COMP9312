import pandas as pd
from sklearn.model_selection import train_test_split



# load the iris dataset and get X and Y data

data = pd.read_csv("satd.csv")
df = pd.DataFrame(data)

train,test = train_test_split(df, test_size=0.20, random_state=0)

train,val = train_test_split(train, test_size=0.10, random_state=0)


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
val.to_csv('val.csv',index=False)

# print(X_train.head())
# print(X_test)
# print(X_valid)

# print(X_train.shape)
# print(X_test.shape)
# print(X_valid.shape)



