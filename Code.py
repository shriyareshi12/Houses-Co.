#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shriya
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr

# load dataset
train = pd.read_csv("/home/shriya/Desktop/train.csv")
test = pd.read_csv("/home/shriya/Desktop/test.csv")

#save the ID column
train_ID = train['Id']
test_ID = test['Id']

#Drop the ID column since it is unnecessary for the prediction process
train.drop("Id",axis =1,inplace = True)
test.drop("Id",axis =1,inplace= True)


print ("Train data: \n")
print ("Number of columns: " + str (train.shape[1]))
print ("number of rows: " + str (train.shape[0]))

print('\nTest data: \n')
print ("number of columns:" + str (test.shape[1]))
print ("Number of rows:" +  str (test.shape[0]))

train.head()

#descriptive statistics summary
train['SalePrice'].describe()

# kernel density plot
'''
In statistics, kernel density estimation (KDE) is a non-parametric way to 
estimate the probability density function of a random variable. 
Kernel density estimation is a fundamental data smoothing problem where 
inferences about the population are made, based on a finite data sample.
'''
sns.distplot(train.SalePrice,fit=norm);
plt.ylabel =('Frequency')
plt.title = ('SalePrice Distribution');
#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice']);
#QQ plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
print("skewness: %f" % train['SalePrice'].skew())
print("kurtosis: %f" % train ['SalePrice'].kurt())

#log transform the target 
'''
SalePrice is right-skewed, and taking the log makes it closer to normally distributed.
'''
train["SalePrice"] = np.log1p(train["SalePrice"])

#Kernel Density plot
sns.distplot(train.SalePrice,fit=norm);
plt.ylabel=('Frequency')
plt.title=('SalePrice distribution');
#Get the fitted parameters used by the function
(mu,sigma)= norm.fit(train['SalePrice']);
#QQ plot
fig =plt.figure()
res =stats. probplot(train['SalePrice'], plot=plt)
plt.show()

print("skewness: %f" % train['SalePrice'].skew())
print("kurtosis: %f" % train ['SalePrice'].kurt())

'''
High kurtosis in a data set is an indicator that data has heavy tails or outliers. 
If there is a high kurtosis, then, we need to investigate why do we have so many outliers.
 It indicates a lot of things, maybe wrong data entry or other things. Investigate!
Low kurtosis in a data set is an indicator that data has light tails or lack of outliers.
 If we get low kurtosis(too good to be true), then also we need to investigate and trim the dataset of unwanted results.

'''

#correration matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,vmax=0.9, square=True)
plt.show()

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

from matplotlib.pyplot import subplots, show



#scatter plot GrLivArea/salePrice
x1 = train['SalePrice']
y1 = train['GrLivArea']

fig, ax = subplots()
ax.scatter(x1, y1)
ax.set_xlabel("SalePrice")
ax.set_ylabel("GrLivArea")
show()

#scatter plot LotArea/salePrice
x2 = train['SalePrice']
y2 = train['LotArea']

fig, ax = subplots()
ax.scatter(x2, y2)
ax.set_xlabel("SalePrice")
ax.set_ylabel("LotArea")
show()

#scatter plot GarageArea/SalePrice
x3 = train['SalePrice']
y3 = train['GarageArea']

fig, ax = subplots()
ax.scatter(x3, y3)
ax.set_xlabel("SalePrice")
ax.set_ylabel("GarageArea")
show()

#scatter plot TotalBsmtSF/SalePrice
x4 = train['SalePrice']
y4 = train['TotalBsmtSF']

fig, ax = subplots()
ax.scatter(x4, y4)
ax.set_xlabel("SalePrice")
ax.set_ylabel("TotalBsmtSF")
show()

#Deleting Outliers of GrLivArea
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train.head()

all_data = pd.concat((train.loc[:, 'MSSubClass': 'SaleCondition'],
                     test.loc[:,'MSSubClass':'SaleCondition']))
print("all_data size is: {} ".format(all_data.shape))
all_data_na = (all_data.isnull().sum()/ len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)
# [:30]
missing_data =pd.DataFrame({'Missing Raio':all_data_na})
missing_data.head(20)

for col in ('PoolQC','MiscFeature','GarageType','Alley','Fence','FireplaceQu','GarageFinish',
           'GarageQual','GarageCond','MasVnrType','MSSubClass'):
    all_data[col] = all_data[col].fillna('None')
    
#Replacing missing value with 0(since no garage = no cars in such garage)
for col in ('GarageYrBlt','GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#missing values are likely zero for no basement 
for col in ('BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
            'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

#
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#for below categorical basement-related feature NaN means that there is no basement 
for col in ('BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
#group by Neigborhood and fill missing value with median Lot frontage of all the neighboorhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
lambda x: x.fillna(x.median()))
    

#msZoning classification: 'RL' is common
all_data ['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#functional: NA is typical
all_data["Functional"] = all_data["Functional"].fillna('Typ')

#Electrical
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#KitchenQual
all_data['KitchenQual'] =all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Extrerior !st and Exterior 2nd
all_data ['Exterior1st']= all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd']= all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#sale type
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

#dropping as same value 'AllPub' for all records except 2NA and 1 'NoSeWa'
all_data = all_data.drop(['Utilities'], axis=1)

#Transforming required numerical features to categorical 
all_data['MSSubClass']= all_data['MSSubClass'].apply(str)
all_data['OverallCond'] =all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#Label Encoding some categorical variables
#for information in their ordering set

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
#apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
#shape
print('Shape all_data: {}'.format(all_data.shape))

#add total surface area as TotalSf = basement + firstflr + secondflr
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#log transform skewed numeric features 
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = all_data[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)
#compute skewness
print ("\skew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})   
print(skewness)

'''
59 Skewed features are :-
----------------------
                    Skew
MiscVal        21.932147
PoolArea       18.701829
LotArea        13.123758
LowQualFinSF   12.080315
3SsnPorch      11.368094
LandSlope       4.971350
KitchenAbvGr    4.298845
BsmtFinSF2      4.142863
EnclosedPorch   4.000796
ScreenPorch     3.943508
BsmtHalfBath    3.942892
MasVnrArea      2.600697
OpenPorchSF     2.529245
WoodDeckSF      1.848285
1stFlrSF        1.253011
LotFrontage     1.092709
GrLivArea       0.977860
BsmtFinSF1      0.974138
TotalSF         0.936173
BsmtUnfSF       0.920135
2ndFlrSF        0.843237
TotRmsAbvGrd    0.749579
Fireplaces      0.725958
HalfBath        0.698770
TotalBsmtSF     0.662657
BsmtFullBath    0.622820
OverallCond     0.569143
HeatingQC       0.484412
FireplaceQu     0.338202
BedroomAbvGr    0.328129
GarageArea      0.217748
OverallQual     0.181902
FullBath        0.159917
MSSubClass      0.141024
YrSold          0.130909
BsmtFinType1    0.082649
GarageCars     -0.219402
YearRemodAdd   -0.449113
BsmtQual       -0.488434
YearBuilt      -0.598087
GarageFinish   -0.611878
LotShape       -0.620602
MoSold         -0.645783
Alley          -0.651370
BsmtExposure   -1.119961
KitchenQual    -1.451569
ExterQual      -1.800172
Fence          -1.994192
ExterCond      -2.495259
BsmtCond       -2.859957
PavedDrive     -2.976397
BsmtFinType2   -3.041630
GarageQual     -3.071423
CentralAir     -3.456087
GarageCond     -3.592789
GarageYrBlt    -3.903046
Functional     -4.052494
Street        -15.489377
PoolQC        -22.984197

* If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
* If the skewness is between -1 and -0.5(negatively skewed) or between 0.5 and 1(positively skewed),
 the data are moderately skewed.
* If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed.

* Positive Skewness means when the tail on the right side of the distribution is longer or fatter. 
    The mean and median will be greater than the mode.
* Negative Skewness is when the tail of the left side of the distribution is longer or fatter than the 
    tail on the right side. The mean and median will be less than the mode.
'''

skewness = skewness[abs(skewness) > 0.75]
print ("There are {} skewed numerical features".format(skewness.shape[0]))

print("\n")
from scipy.special import boxcox1p 
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    

all_data = pd.get_dummies(all_data)
#print(all_data.shape)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train= train.SalePrice.values
train = pd.DataFrame(all_data[:ntrain])
test = pd.DataFrame(all_data[ntrain:])

from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
'''
KFold ->
    K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. 
    Split dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
    
cross_val_score ->
    estimator estimator object implementing ‘fit’
    The object to use to fit the data.

    Xarray-like of shape (n_samples, n_features)
    The data to fit. Can be for example a list, or an array.

    yarray-like of shape (n_samples,) or (n_samples, n_outputs), default=None
    The target variable to try to predict in the case of supervised learning.

    groupsarray-like of shape (n_samples,), default=None
    Group labels for the samples used while splitting the dataset into train/test set.
    Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).

    scoring str or callable, default=None
    A str (see model evaluation documentation) or a scorer callable object /
    function with signature scorer(estimator, X, y) which should return only a single value.  
'''
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#validation function
n_folds = 5

'''
**n_splits int, default=5
    Number of folds. Must be at least 2.

**shufflebool, default=False
    Whether to shuffle the data before splitting into batches.
    Note that the samples within each split will not be shuffled.
    
**random_state int or RandomState instance, default=None
    When shuffle is True, random_state affects the ordering of the indices,
    which controls the randomness of each fold. Otherwise, this parameter has no effect.
'''

def RMSLE_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=40).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error",
cv = kf))
    return(rmse)
    

#lasso
'''
**make_pipeline
    Construct a Pipeline from the given estimators.

**RobustScaler()
    Scale features using statistics that are robust to outliers.

**alphafloat, default=1.0
    Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent
    to an ordinary least square, solved by the LinearRegression object. 
    For numerical reasons, using alpha = 0 with the Lasso object is not advised.

**random_stateint, RandomState instance, default=None
    The seed of the pseudo random number generator that selects a random feature to update. 
'''
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))

#Gradient Boosting Regression
'''
**loss{‘ls’, ‘lad’, ‘huber’, ‘quantile’}, default=’ls’
    loss function to be optimized. ‘ls’ refers to least squares regression. 
    ‘lad’ (least absolute deviation) is a highly robust loss function solely
    based on order information of the input variables. ‘huber’ is a combination
    of the two. ‘quantile’ allows quantile regression (use alpha to specify the quantile).

**learning_ratefloat, default=0.1
    learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.

**n_estimator — The number of boosting stages to perform. We should not set it too high which would overfit our model.

**max_depth — The depth of the tree node.

**learning_rate — Rate of learning the data.

**loss — loss function to be optimized. ‘ls’ refers to least squares regression

**minimum sample split — Number of sample to be split for learning the data

**max_features{‘auto’, ‘sqrt’, ‘log2’}, int or float, default=None
    The number of features to consider when looking for the best split.
    If “sqrt”, then max_features=sqrt(n_features)
    Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
'''
GBoost = GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=3000,
                                   min_samples_split=10, min_samples_leaf=15,max_depth=4,
                                   random_state=5,max_features='sqrt')


#Lasso
score = RMSLE_cv(lasso)
print ("\n Lasso score: {:.4f} ({:.4f})\n".format(score.mean(),score.std()))

#Gradient Boosting Regression
score = RMSLE_cv(GBoost)
print ("\n GBoost score: {:.4f} ({:.4f})\n".format(score.mean(),score.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


    
# Averaged base models score

averaged_models = AveragingModels(models = (GBoost, lasso))

score = RMSLE_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#defining RMSLE evaluation function
def RMSLE (y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#final training and prediction of the stacked regressor

averaged_models.fit(train.values, y_train) 
stacked_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print("RMSLE score on the train data:") 
print(RMSLE(y_train,stacked_train_pred))
print("Accuracy score:") 
print(averaged_models.score(train.values, y_train))

from sqlalchemy import create_engine
import mysql.connector
engine = create_engine("mysql+mysqlconnector://root:tiger@localhost/prediction", echo=False)
con = engine.connect()
    

ensemble = stacked_pred *1
submit = pd.DataFrame()
submit['id'] = test_ID
submit['SalePrice'] = ensemble
submit.to_sql(con=con, name = 'submission', if_exists='replace', index=False)
print(submit)
print(submit.describe())
con.close()


from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure
import matplotlib.patches

mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    password="tiger",
    database="prediction"
)
mycursor=mydb.cursor()

root=Tk()
root.title("Housing")
root.configure(width=2500,height=600,bg="Grey")
photo=tk.PhotoImage(file='/home/shriya/Desktop/qwe.png')
label12=Label(root,image=photo)
label12.place(x=0, y=0, relwidth=1, relheight=1)

path = '/home/shriya/Desktop/abc.png'
frame2 = tk.PhotoImage(file=path)
label13=Label(root,image=frame2)
label13.grid(row=1, column=2)

import time
 
time1 = ''
clock = Label(root, font=('times', 20, 'bold'), bg='white')
#clock.pack(fill=BOTH, expand=1)
clock.grid(row=0,column=0, padx=5, pady=5) 
def tick():
    global time1
    # get the current local time from the PC
    time2 = time.strftime('%H:%M:%S')
    # if time string has changed, update it
    if time2 != time1:
        time1 = time2
        clock.config(text=time2)
    # calls itself every 200 milliseconds
    # to update the time display as needed
    # could use >200 ms, but display gets jerky
    clock.after(200, tick)

 
tick()


#label0= Label(root,text="Awas",bg="Black",fg="#F9FAE9",font=("Times", 60))
label1=Label(root,text="House Id",bg="black",relief="ridge",fg="white",bd=8,font=("Times", 12),width=25)
entry1=Entry(root , font=("Times", 14),bd=8,width=15,bg="white")
label2=Label(root, text="Sale Price",relief="ridge",height="1",bg="black",bd=8,fg="white", font=("Times", 12),width=25)
entry2= Entry(root, font=("Times", 14),bd=8,width=15,bg="white")
entry3= Entry(root, font=("Times", 14),bd=8,width=25,bg="white")
    

def displayGraph():
    root = Tk()
    root.configure(bg="White")
    root.title("Housing Database")
    
    def back():
        root.destroy()
        return
    
    train = pd.read_csv("/home/shriya/Desktop/train.csv")

    df3 = pd.DataFrame(train, columns = ['SalePrice', 'GrLivArea'])
    
    label01= Label(root,text="Visualizations",bg="Black",fg="#F9FAE9",font=("Times", 40))
    label01.grid(row=0,column=1, padx=10, pady=10)
    
    button8= Button(root,activebackground="green", text="BACK",bd=8, bg=buttoncolor, fg=buttonfg, width =4, font=("Times", 12),command=back)
    button8.grid(row=2,column=2, padx=10, pady=10)
    
    figure3 = plt.Figure(figsize=(8,3), dpi=80)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df3['SalePrice'],df3['GrLivArea'], color = 'g')
    scatter3 = FigureCanvasTkAgg(figure3, root) 
    scatter3.get_tk_widget().grid(row=1,column=1, padx=10, pady=10)
    ax3.legend(['SalePrice']) 
    ax3.set_title('SalePrice Vs. GrLivArea')
 
    df4 = pd.DataFrame(train, columns = ['SalePrice', 'GarageArea'])
    
    figure4 = plt.Figure(figsize=(8,3), dpi=80)
    ax4 = figure4.add_subplot(111)
    ax4.scatter(df4['SalePrice'],df4['GarageArea'], color = 'g')
    scatter4 = FigureCanvasTkAgg(figure4, root) 
    scatter4.get_tk_widget().grid(row=1,column=2, padx=10, pady=10)
    ax4.legend(['SalePrice']) 
    ax4.set_title('SalePrice Vs. GarageArea')
    
    df5 = pd.DataFrame(train, columns = ['SalePrice', 'OverallQual'])
    
    figure5 = plt.Figure(figsize=(8,3), dpi=80)
    ax5 = figure5.add_subplot(111)
    ax5.scatter(df5['SalePrice'],df5['OverallQual'], color = 'g')
    scatter5 = FigureCanvasTkAgg(figure5, root) 
    scatter5.get_tk_widget().grid(row=2,column=1, padx=10, pady=10)
    ax5.legend(['SalePrice']) 
    ax5.set_title('SalePrice Vs. OverallQual')
    
   
def qExit():
    qExit= messagebox.askyesno("Quit System","Do you want to quit? \n \n Thank You.....!!")
    if qExit > 0:
        root.destroy()
        return 
    

def showdatabase():
        root = Tk()
        root.configure(bg="Grey")
        root.title("Housing Database")
        root.resizable(width = 1, height = 1) 
        tv= ttk.Treeview(root)
        tv.pack(expand=True, fill='y')
        
        tv['columns']=('Id' , 'Street' , 'Alley' , 'Utilities' , 'Neighborhood' , 'BldgType' , 'HouseStyle' , 'OverallQual' , 'OverallCond' , 'YearBuilt')
        tv.heading('#0',text='Id',anchor='center')
        tv.column('#0',minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#1', text='Street', anchor='center')
        tv.column('#1', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#2', text='Alley', anchor='center')
        tv.column('#2', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#3', text='Utilities', anchor='center')
        tv.column('#3', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#4', text='Neighborhood', anchor='center')
        tv.column('#4', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#5', text='BldgType', anchor='center')
        tv.column('#5', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#6', text='HouseStyle', anchor='center')
        tv.column('#6', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#7', text='OverallQual', anchor='center')
        tv.column('#7', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#8', text='OverallCond', anchor='center')
        tv.column('#8', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.heading('#9', text='YearBuilt', anchor='center')
        tv.column('#9', minwidth=0, width=130, stretch=NO, anchor='center')
        tv.tag_configure('gray', background='#cccccc')
        
        Select="select Id, Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt from test"
        mycursor.execute(Select)
        result=mycursor.fetchall()
        Id=""
        Street=""
        Alley=""
        Utilities=""
        Neighborhood=""
        BldgType=""
        HouseStyle=""
        OverallQual=""
        OverallCond=""
        YearBuilt=""
        for i in result:
            Id=i[0]
            Street=i[1]
            Alley=i[2]
            Utilities=i[3]
            Neighborhood=i[4]
            BldgType=i[5]
            HouseStyle=i[6]
            OverallQual=i[7]
            OverallCond=i[8]
            YearBuilt=i[9]
            tv.insert("",'end',text=Id,values=(Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt))
        
             
def searchData():
        e6 = entry1.get()
        mycursor.execute("select SalePrice from submission where id = '%s'"%e6)
        mytext2 = mycursor.fetchall()
        entry2.insert(END,mytext2[0])
        
        
def Clear():
        entry1.delete(0, END)
        entry2.delete(0, END)
        
def DashBoard():
    root= Tk()
    root.configure(bg="White")
    
    fig1 = matplotlib.figure.Figure(figsize=(2,2))
    ax = fig1.add_subplot(111)
    ax.pie([73,27]) 
    ax.legend(["73"])
    ax.set_title('Lasso Accuracy')
    
    
    circle=matplotlib.patches.Circle( (0,0), 0.7, color='white')
    ax.add_artist(circle)
    a5 = FigureCanvasTkAgg(fig1, root)
    a5.get_tk_widget().grid(row=1,column=2, padx=10, pady=10)
    
    fig2 = matplotlib.figure.Figure(figsize=(2,2))
    ax1 = fig2.add_subplot(111)
    ax1.pie([73,27]) 
    ax1.legend(["75"])
    ax1.set_title('GBoost Accuracy')
    
    
    circle=matplotlib.patches.Circle( (0,0), 0.7, color='white')
    ax1.add_artist(circle)
    a6 = FigureCanvasTkAgg(fig2, root)
    a6.get_tk_widget().grid(row=1,column=4, padx=10, pady=10)
    
    
    label14=Label(root,text="\nTotal \n\n 1459 \n\n Properties\n ",bg="black",relief="ridge",fg="white",bd=8,font=("Helvetica", 10),width=23)
    label14.grid(row=1, column=6)
    
    label15=Label(root,text="\n MOST EXPENSIVE : \n\n $  686461.806 \n\n Neighborhood : Edwards \n ",bg="black",relief="ridge",fg="white",bd=8,font=("Helvetica", 10),width=23)
    label15.grid(row=2, column=6)
    
    label16=Label(root,text="\n LEAST EXPENSIVE : \n\n $ 48440.697 \n\n Neighborhood : IDOTRR \n ",bg="black",relief="ridge",fg="white",bd=8,font=("Helvetica", 10),width=23)
    label16.grid(row=3, column=6)
    
    train = pd.read_csv("/home/shriya/Desktop/train.csv")

    
    df2 = pd.DataFrame(train, columns = ['HouseStyle' , 'SalePrice'])
    
    figure2 = plt.Figure(figsize=(9,4), dpi=60)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, root)
    line2.get_tk_widget().grid(row=3,column=4, padx=10, pady=10)
    df2 = df2[['HouseStyle','SalePrice']].groupby('HouseStyle').sum()
    df2.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=10)
    ax2.set_title('HouseStyle Vs. SalePrice')

    
    df3 = pd.DataFrame(train, columns = ['YearBuilt' , 'SalePrice'])
    
    figure3 = plt.Figure(figsize=(9,4), dpi=60)
    ax3 = figure3.add_subplot(111)
    line3 = FigureCanvasTkAgg(figure3, root)
    line3.get_tk_widget().grid(row=2,column=4, padx=10, pady=10)
    df3 = df3[['YearBuilt','SalePrice']].groupby('YearBuilt').sum()
    df3.plot(kind='line', legend=True, ax=ax3, color='r',marker='o', fontsize=10)
    ax3.set_title('YearBuilt Vs. SalePrice')
    
    df4 = pd.DataFrame(train, columns = ['BldgType' , 'SalePrice'])
    
    figure4 = plt.Figure(figsize=(9,4), dpi=60)
    ax4 = figure4.add_subplot(111)
    line4 = FigureCanvasTkAgg(figure4, root)
    line4.get_tk_widget().grid(row=3,column=2, padx=10, pady=10)
    df4 = df4[['BldgType','SalePrice']].groupby('BldgType').sum()
    df4.plot(kind='line', legend=True, ax=ax4, color='r',marker='o', fontsize=10)
    ax4.set_title('BldgType Vs. SalePrice')
   
    
    df5 = pd.DataFrame(train, columns = ['Neighborhood' , 'SalePrice'])
    
    figure5 = plt.Figure(figsize=(9,4), dpi=60)
    ax5 = figure5.add_subplot(111)
    line5 = FigureCanvasTkAgg(figure5, root)
    line5.get_tk_widget().grid(row=2,column=2, padx=10, pady=10)
    df5 = df5[['Neighborhood','SalePrice']].groupby('Neighborhood').sum()
    df5.plot(kind='line', legend=True, ax=ax5, color='r',marker='o', fontsize=10)
    ax5.set_title('Neighborhood Vs. SalePrice')
   
    
def filterData():
        root = Tk()
        root.configure(bg="Grey")
        root.title("Housing Database")
        def filterBldgType():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            root.resizable(width = 1, height = 1) 
            tv= ttk.Treeview(root)
            tv.pack(expand=True, fill='y')
            
            tv['columns']=('Id' , 'Street' , 'Alley' , 'Utilities' , 'Neighborhood' , 'BldgType' , 'HouseStyle' , 'OverallQual' , 'OverallCond' , 'YearBuilt')
            tv.heading('#0',text='Id',anchor='center')
            tv.column('#0',minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#1', text='Street', anchor='center')
            tv.column('#1', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#2', text='Alley', anchor='center')
            tv.column('#2', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#3', text='Utilities', anchor='center')
            tv.column('#3', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#4', text='Neighborhood', anchor='center')
            tv.column('#4', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#5', text='BldgType', anchor='center')
            tv.column('#5', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#6', text='HouseStyle', anchor='center')
            tv.column('#6', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#7', text='OverallQual', anchor='center')
            tv.column('#7', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#8', text='OverallCond', anchor='center')
            tv.column('#8', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#9', text='YearBuilt', anchor='center')
            tv.column('#9', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.tag_configure('gray', background='#cccccc')
            
            i1 = entry3.get()
            mycursor.execute("select Id, Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt from test where BldgType = '%s'"%i1)
            result=mycursor.fetchall()
            Id=""
            Street=""
            Alley=""
            Utilities=""
            Neighborhood=""
            BldgType=""
            HouseStyle=""
            OverallQual=""
            OverallCond=""
            YearBuilt=""
            for i in result:
                Id=i[0]
                Street=i[1]
                Alley=i[2]
                Utilities=i[3]
                Neighborhood=i[4]
                BldgType=i[5]
                HouseStyle=i[6]
                OverallQual=i[7]
                OverallCond=i[8]
                YearBuilt=i[9]
                tv.insert("",'end',text=Id,values=(Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt))
            
        
        def filterNeighborhood():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            root.resizable(width = 1, height = 1) 
            tv= ttk.Treeview(root)
            tv.pack(expand=True, fill='y')
            
            tv['columns']=('Id' , 'Street' , 'Alley' , 'Utilities' , 'Neighborhood' , 'BldgType' , 'HouseStyle' , 'OverallQual' , 'OverallCond' , 'YearBuilt')
            tv.heading('#0',text='Id',anchor='center')
            tv.column('#0',minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#1', text='Street', anchor='center')
            tv.column('#1', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#2', text='Alley', anchor='center')
            tv.column('#2', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#3', text='Utilities', anchor='center')
            tv.column('#3', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#4', text='Neighborhood', anchor='center')
            tv.column('#4', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#5', text='BldgType', anchor='center')
            tv.column('#5', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#6', text='HouseStyle', anchor='center')
            tv.column('#6', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#7', text='OverallQual', anchor='center')
            tv.column('#7', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#8', text='OverallCond', anchor='center')
            tv.column('#8', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#9', text='YearBuilt', anchor='center')
            tv.column('#9', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.tag_configure('gray', background='#cccccc')
            
            i1 = entry3.get()
            mycursor.execute("select Id, Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt from test where Neighborhood = '%s'"%i1)
            result=mycursor.fetchall()
            Id=""
            Street=""
            Alley=""
            Utilities=""
            Neighborhood=""
            BldgType=""
            HouseStyle=""
            OverallQual=""
            OverallCond=""
            YearBuilt=""
            for i in result:
                Id=i[0]
                Street=i[1]
                Alley=i[2]
                Utilities=i[3]
                Neighborhood=i[4]
                BldgType=i[5]
                HouseStyle=i[6]
                OverallQual=i[7]
                OverallCond=i[8]
                YearBuilt=i[9]
                tv.insert("",'end',text=Id,values=(Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt))
            
        def filterBuget():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            root.resizable(width = 1, height = 1) 
            tv= ttk.Treeview(root)
            tv.pack(expand=True, fill='y')
            
            tv['columns']=('Id' , 'SalePrice' ,'Street' , 'Utilities' , 'Neighborhood' , 'BldgType' , 'HouseStyle' , 'OverallQual' , 'OverallCond' , 'YearBuilt')
            tv.heading('#0',text='Id',anchor='center')
            tv.column('#0',minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#1', text='SalePrice', anchor='center')
            tv.column('#1', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#2', text='Street', anchor='center')
            tv.column('#2', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#3', text='Utilities', anchor='center')
            tv.column('#3', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#4', text='Neighborhood', anchor='center')
            tv.column('#4', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#5', text='BldgType', anchor='center')
            tv.column('#5', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#6', text='HouseStyle', anchor='center')
            tv.column('#6', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#7', text='OverallQual', anchor='center')
            tv.column('#7', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#8', text='OverallCond', anchor='center')
            tv.column('#8', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#9', text='YearBuilt', anchor='center')
            tv.column('#9', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.tag_configure('gray', background='#cccccc')
            
            i1 = entry3.get()
            mycursor.execute("SELECT submission.id, submission.SalePrice, test.Street, test.Utilities, Neighborhood, test.BldgType, test.HouseStyle, test.OverallQual, test.OverallCond, test.YearBuilt FROM submission INNER JOIN test on submission.id=test.id WHERE SalePrice <= '%s'"%i1)
            result=mycursor.fetchall()
            Id=""
            SalePrice=""
            Street=""
            Utilities=""
            Neighborhood=""
            BldgType=""
            HouseStyle=""
            OverallQual=""
            OverallCond=""
            YearBuilt=""
            for i in result:
                Id=i[0]
                SalePrice=i[1]
                Street=i[2]
                Utilities=i[3]
                Neighborhood=i[4]
                BldgType=i[5]
                HouseStyle=i[6]
                OverallQual=i[7]
                OverallCond=i[8]
                YearBuilt=i[9]
                tv.insert("",'end',text=Id,values=(SalePrice, Street, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt))
            
        
        def filterYearBlt():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            root.resizable(width = 1, height = 1) 
            tv= ttk.Treeview(root)
            tv.pack(expand=True, fill='y')
            
            tv['columns']=('Id' , 'SalePrice' , 'Street' , 'Utilities' , 'Neighborhood' , 'BldgType' , 'HouseStyle' , 'OverallQual' , 'OverallCond' , 'YearBuilt')
            tv.heading('#0',text='Id',anchor='center')
            tv.column('#0',minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#1', text='SalePrice', anchor='center')
            tv.column('#1', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#2', text='Street', anchor='center')
            tv.column('#2', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#3', text='Utilities', anchor='center')
            tv.column('#3', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#4', text='Neighborhood', anchor='center')
            tv.column('#4', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#5', text='BldgType', anchor='center')
            tv.column('#5', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#6', text='HouseStyle', anchor='center')
            tv.column('#6', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#7', text='OverallQual', anchor='center')
            tv.column('#7', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#8', text='OverallCond', anchor='center')
            tv.column('#8', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#9', text='YearBuilt', anchor='center')
            tv.column('#9', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.tag_configure('gray', background='#cccccc')
            
            i1 = entry3.get()
            mycursor.execute("SELECT submission.id, submission.SalePrice, test.Street, test.Utilities, Neighborhood, test.BldgType, test.HouseStyle, test.OverallQual, test.OverallCond, test.YearBuilt FROM submission INNER JOIN test on submission.id=test.id WHERE YearBuilt = '%s'"%i1)
            result=mycursor.fetchall()
            Id=""
            SalePrice=""
            Street=""
            Utilities=""
            Neighborhood=""
            BldgType=""
            HouseStyle=""
            OverallQual=""
            OverallCond=""
            YearBuilt=""
            for i in result:
                Id=i[0]
                SalePrice=i[1]
                Street=i[2]
                Utilities=i[3]
                Neighborhood=i[4]
                BldgType=i[5]
                HouseStyle=i[6]
                OverallQual=i[7]
                OverallCond=i[8]
                YearBuilt=i[9]
                tv.insert("",'end',text=Id,values=(SalePrice, Street, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt))
            
            
        def filterGrLivArea():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            root.resizable(width = 1, height = 1) 
            tv= ttk.Treeview(root)
            tv.pack(expand=True, fill='y')
            
            tv['columns']=('Id' , 'SalePrice' , 'Street' , 'GrLivArea' , 'Neighborhood' , 'BldgType' , 'HouseStyle' , 'OverallQual' , 'OverallCond' , 'YearBuilt')
            tv.heading('#0',text='Id',anchor='center')
            tv.column('#0',minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#1', text='SalePrice', anchor='center')
            tv.column('#1', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#2', text='Street', anchor='center')
            tv.column('#2', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#3', text='GrLivArea', anchor='center')
            tv.column('#3', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#4', text='Neighborhood', anchor='center')
            tv.column('#4', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#5', text='BldgType', anchor='center')
            tv.column('#5', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#6', text='HouseStyle', anchor='center')
            tv.column('#6', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#7', text='OverallQual', anchor='center')
            tv.column('#7', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#8', text='OverallCond', anchor='center')
            tv.column('#8', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.heading('#9', text='YearBuilt', anchor='center')
            tv.column('#9', minwidth=0, width=130, stretch=NO, anchor='center')
            tv.tag_configure('gray', background='#cccccc')
            
            i1 = entry3.get()
            mycursor.execute("SELECT submission.id, submission.SalePrice, test.Street, test.GrLivArea, Neighborhood, test.BldgType, test.HouseStyle, test.OverallQual, test.OverallCond, test.YearBuilt FROM submission INNER JOIN test on submission.id=test.id WHERE GrLivArea = '%s'"%i1)
            result=mycursor.fetchall()
            Id=""
            SalePrice=""
            Street=""
            GrLivArea=""
            Neighborhood=""
            BldgType=""
            HouseStyle=""
            OverallQual=""
            OverallCond=""
            YearBuilt=""
            for i in result:
                Id=i[0]
                SalePrice=i[1]
                Street=i[2]
                GrLivArea=i[3]
                Neighborhood=i[4]
                BldgType=i[5]
                HouseStyle=i[6]
                OverallQual=i[7]
                OverallCond=i[8]
                YearBuilt=i[9]
                tv.insert("",'end',text=Id,values=(SalePrice, Street, GrLivArea, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt))
            
        
        
        entry3= Entry(root, font=("Times", 14),bd=8,width=25,bg="white")
        button4= Button(root,activebackground="green", text="Filter by Building Type",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterBldgType)
        button5= Button(root,activebackground="green", text="Filter by Neighborhood",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterNeighborhood)
        button14= Button(root,activebackground="green", text="Filter by Budget",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterBuget)
        button15= Button(root,activebackground="green", text="Filter by Year Built",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterYearBlt)
        button16= Button(root,activebackground="green", text="Filter by Area",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterGrLivArea)

        
        entry3.grid(row=3,column=4, padx=10, pady=10)
        button4.grid(row=4,column=4, padx=10, pady=10)
        button5.grid(row=5,column=4, padx=10, pady=10)
        button14.grid(row=6,column=4, padx=10, pady=10)
        button15.grid(row=7,column=4, padx=10, pady=10)
        button16.grid(row=8,column=4, padx=10, pady=10)


   
        

buttoncolor="black"
buttonfg="white"
button1= Button(root,activebackground="green", text="SEARCH Predited price",bd=8, bg=buttoncolor, fg=buttonfg, width=25, font=("Times", 12),command=searchData)
button2= Button(root,activebackground="green", text="CLEAR",bd=8, bg=buttoncolor, fg=buttonfg, width =15, font=("Times", 12),command=Clear)
button3= Button(root,activebackground="green", text="VIEW PROPERTIES",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=showdatabase)
button5= Button(root,activebackground="green", text="FILTER",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterData)
button20= Button(root,activebackground="green", text="DASHBOARD",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=DashBoard)
button6= Button(root,activebackground="green", text="SHOW ANALYSIS",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=displayGraph)
button7= Button(root,activebackground="green", text="EXIT",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=qExit)
button20= Button(root,activebackground="green", text="DASHBOARD",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=DashBoard)
  
    
#label0.grid(row=1,column=4, padx=10, pady=10)    
label1.grid(row=6,column=0, padx=10, pady=10)
label2.grid(row=7,column=0, padx=10, pady=10)
    
entry1.grid(row=6,column=1, padx=10, pady=10)
entry2.grid(row=7,column=1, padx=10, pady=10)
    
button1.grid(row=8,column=1, padx=10, pady=10)
button2.grid(row=9,column=1, padx=10, pady=10)
button3.grid(row=6,column=6, padx=10, pady=10)
button5.grid(row=7,column=6, padx=10, pady=10)
button6.grid(row=9,column=6, padx=10, pady=10)
button7.grid(row=12,column=6, padx=10, pady=10)
button20.grid(row=8,column=6, padx=10, pady=10)


root.mainloop()     
