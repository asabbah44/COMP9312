import pandas as pd
from sklearn.model_selection import train_test_split


# load the iris dataset and get X and Y data
 #["DESIGN", "IMPLEMENTATION", "TEST","DEFECT","DOCUMENTATION"],
data = pd.read_csv("TDSub1.csv")
df = pd.DataFrame(data)
df_skip_doc=df[df.label == 'DOCUMENTATION']
df_skip_test=df[df.label == 'TEST']

df_in_design=df[df.label == 'DESIGN']
df_in_implementation=df[df.label == 'IMPLEMENTATION']
df_in_defect=df[df.label == 'DEFECT']

IncludeDesign,ExcludeDes = train_test_split(df_in_design, test_size=0.90, random_state=0)
IncludeImplementation,ExcludeImp = train_test_split(df_in_implementation, test_size=0.60, random_state=0)
IncludeDef,ExcludeDef = train_test_split(df_in_defect, test_size=0.50, random_state=0)


df=pd.concat([IncludeDesign, IncludeImplementation,IncludeDef, df_skip_doc, df_skip_test])
print(df.count())

# train,test = train_test_split(df, test_size=0.20, random_state=0)
#
# train,val = train_test_split(train, test_size=0.10, random_state=0)
#
#
# train.to_csv('train.csv',index=False)
# test.to_csv('test.csv',index=False)
# val.to_csv('val.csv',index=False)

# print(X_train.head())
# print(X_test)
# print(X_valid)

# print(X_train.shape)
# print(X_test.shape)
# print(X_valid.shape)



