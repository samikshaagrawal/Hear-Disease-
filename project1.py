import pydbgen
from pydbgen import pydbgen
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Generation Of Synthetic Dataset

myDB = pydbgen.pydb()
testdf = myDB.gen_dataframe(843, fields=['name','city', 'phone', 'ssn'])
testdf.columns= ['name','city', 'phone', 'ssn']

#Supressing the sensitive attribute

testdf.name='****'
testdf.city='USA'
testdf.phone='**********'
testdf["ssn"]=testdf["ssn"].apply(lambda x: "***-**" + x[6:])

#Importing Heart Dataset from UCI repository

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
         'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
hd = pd.read_csv('cleavecol.csv', header = None, names=names) 
hd['chol'] = hd['chol'].replace('?',0)

#Preprocessing of Cholestrol attribute of heart dataset
hd.loc[(hd['age'] <=40) & (hd['sex'] == 0) & (hd['chol'] == 0),'chol'] =119
hd.loc[(hd['age'] > 40) & (hd['age'] < 50) & (hd['sex'] == 0) &(hd['chol'] == 0),'chol'] =183
hd.loc[(hd['age'] >=50) & (hd['sex'] == 0) & (hd['chol'] == 0),'chol'] =219
hd.loc[(hd['age'] <=40) & (hd['sex'] == 1) & (hd['chol'] == 0),'chol'] =185
hd.loc[(hd['age'] > 40) & (hd['age'] < 50) & (hd['sex'] == 1) &(hd['chol'] == 0),'chol'] =205
hd.loc[(hd['age'] >=50) & (hd['sex'] == 1) & (hd['chol'] == 0),'chol'] =208
hd.chol.dtype
hd.chol=pd.to_numeric(hd.chol)
for i in range(1,5):
    hd['heartdisease'] = hd['heartdisease'].replace(i,1)

#Binning to convert numerical attributes to categorical form


data = hd.chol
bins = np.linspace(100, 600, 5)
hd.chol = np.digitize(data, bins)
bin_means = [data[hd.chol == i].mean() for i in range(1, len(bins))]

data = hd.trestbps
bins = np.linspace(90, 200, 5)
hd.trestbps = np.digitize(data, bins)
bin_means = [data[hd.trestbps == i].mean() for i in range(1, len(bins))]

data = hd.thalach
bins = np.linspace(90, 200, 5)
hd.thalach = np.digitize(data, bins)
bin_means = [data[hd.thalach == i].mean() for i in range(1, len(bins))]

data = hd.oldpeak
bins = np.linspace(0, 6, 5)
hd.oldpeak = np.digitize(data, bins)
bin_means = [data[hd.oldpeak == i].mean() for i in range(1, len(bins))]

data = hd.age
bins = np.linspace(20, 80, 5)
hd.age = np.digitize(data, bins)
bin_means = [data[hd.age == i].mean() for i in range(1, len(bins))]

#importing the smoke dataset
from statistics import median
list=['res','nos','year']
smoke=pd.read_csv('smoke.csv',header=None,names=list)

#Handling outliers ofsmoke dataset
median1 = smoke.loc[smoke['year']<=0, 'year'].median()
smoke.loc[smoke.year<0, 'year'] = np.nan
smoke.fillna(median1,inplace=True)
median2 = smoke.loc[smoke['nos']<=0, 'nos'].median()
smoke.loc[smoke.nos<0, 'nos'] = np.nan
smoke.fillna(median2,inplace=True)

#Preprocessing of smoke dataset
smoke.loc[(smoke['nos'] ==0) & (smoke['year'] == 0),'res'] =0
smoke.loc[(smoke['nos'] >0) & (smoke['year'] >0),'res'] =1

#Binning to convert numerical attributes to categorical form
data = smoke.nos
bins = np.linspace(0, 80, 5)
smoke.nos = np.digitize(data, bins)
bin_means = [data[smoke.nos == i].mean() for i in range(1, len(bins))]

data = smoke.year
bins = np.linspace(0, 60, 5)
smoke.year = np.digitize(data, bins)
bin_means = [data[smoke.year == i].mean() for i in range(1, len(bins))]

#Concatenating all the three datasets
hd=hd.replace('?',0.0)
df_col_merged =pd.concat([testdf, hd], axis=1)
final =pd.concat([df_col_merged, smoke], axis=1)
final=final.replace('?',0.0)

#Classifying attributes into dependent and independent
X=final.iloc[:,[4,5,6,7,8,9,10,11,12,13,14,15,16]].values
y=final.iloc[:,17].values
dfx=pd.DataFrame(X)

#Spliting the dataset into train and test set
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.10,random_state=42)
dfxtest=pd.DataFrame(Xtest)
dfxtrain=pd.DataFrame(Xtrain)

#Applying the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(max_depth=5,random_state=0)
classifier.fit(Xtrain,ytrain)
ypred=classifier.predict(Xtest)

#Accuracy analysis of decision tree classifier 
print('accuracy on training subset:{:.3f} decision tree'.format(classifier.score(Xtrain,
ytrain)))
print('accuracy on testing subset:{:.3f} decision tree'.format(classifier.score(Xtest,
ytest)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)

#Applying the random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1,criterion='entropy',random_state=99)
classifier.fit(Xtrain,ytrain)
ypred=classifier.predict(Xtest)

#Accuracy analysis of random forest classifier 
print('accuracy on training subset:{:.3f} random forest'.format(classifier.score(Xtrain,
ytrain)))
print('accuracy on testing subset:{:.3f} random forest'.format(classifier.score(Xtest,
ytest)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)








