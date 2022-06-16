import pandas as pd
from sklearn.model_selection import train_test_split



# load the iris dataset and get X and Y data

data = pd.read_csv("train.csv")
df = pd.DataFrame(data)

train, skip = train_test_split(df, test_size=0.000001, random_state=0)

#train,val = train_test_split(train, test_size=0.10, random_state=0)

print(f"Training target statistics: ",train.count())
from imblearn.over_sampling import RandomOverSampler,SMOTEN

x=pd.DataFrame(train["text"])
y=pd.DataFrame(train["label"])

over_sampler = RandomOverSampler(random_state=42)
X_res, y_res = over_sampler.fit_resample(x, y)

SMOTEN_over_sampler = SMOTEN(random_state=42)
X_res_smo, y_res_som = SMOTEN_over_sampler.fit_resample(x, y)

print(f"Training target Random: ",X_res.count())
print(f"Training target  SMOTEN : ",X_res_smo.count())

trainOver=pd.concat([X_res, y_res], axis=1)
trainSMOTEN=pd.concat([X_res_smo, y_res_som], axis=1)
print(trainOver.head())

trainOver.to_csv('R_overtrain.csv',index=False)
trainSMOTEN.to_csv('SMOTEN_train.csv',index=False)
# val.to_csv('val.csv',index=False)

# print(X_train.head())
# print(X_test)
# print(X_valid)

# print(X_train.shape)
# print(X_test.shape)
# print(X_valid.shape)



