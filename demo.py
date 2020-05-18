#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:04:56 2020

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
print ("Number of columns:" +  str (test.shape[0]))

train.head()

#descriptive statistics summary
train['SalePrice'].describe()

# kernel density plot
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

var ='TotalBsmtSF'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim =0.80000);
plt.show()

#scatter plot LotArea/salePrice
var = 'LotArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x= var, y='SalePrice', ylim =(0,50));
plt.show()

#scatter plot GrLivArea/salePrice
var ='GrLivArea'
data =pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice',ylim=(0,50));
plt.show()

#scatter plot GarageArea/SalePrice
var = 'GarageArea'
data =pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var,y='SalePrice', ylim= (3,50));
plt.show()

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
skewness.head(7)   

skewness = skewness[abs(skewness) > 0.75]
print ("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p 
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    

all_data = pd.get_dummies(all_data)
print(all_data.shape)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train= train.SalePrice.values
train = pd.DataFrame(all_data[:ntrain])
test = pd.DataFrame(all_data[ntrain:])

from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#validation function
n_folds = 5

def RMSLE_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error",
cv = kf))
    return(rmse)
    

#lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))

#Gradient Boosting Regression
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
from tkinter import messagebox
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    root.configure(bg="Grey")
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
    
    figure3 = plt.Figure(figsize=(6,3), dpi=100)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df3['SalePrice'],df3['GrLivArea'], color = 'g')
    scatter3 = FigureCanvasTkAgg(figure3, root) 
    scatter3.get_tk_widget().grid(row=1,column=1, padx=10, pady=10)
    ax3.legend(['SalePrice']) 
    ax3.set_title('SalePrice Vs. GrLivArea')
 
    df4 = pd.DataFrame(train, columns = ['SalePrice', 'GarageArea'])
    
    figure4 = plt.Figure(figsize=(6,3), dpi=100)
    ax4 = figure4.add_subplot(111)
    ax4.scatter(df4['SalePrice'],df4['GarageArea'], color = 'g')
    scatter4 = FigureCanvasTkAgg(figure4, root) 
    scatter4.get_tk_widget().grid(row=1,column=2, padx=10, pady=10)
    ax4.legend(['SalePrice']) 
    ax4.set_title('SalePrice Vs. GarageArea')
    
    df5 = pd.DataFrame(train, columns = ['SalePrice', 'OverallQual'])
    
    figure5 = plt.Figure(figsize=(7,3), dpi=100)
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
        mycursor.execute("select Id, Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt from test")
        mytext1 = mycursor.fetchall()
        mytext = Text(root,width=500,height= 200 ,bg= "gray",fg="black", font=("Times", 12))
        mytext.insert(END," Id \t\t Street \t\t Alley \t\t Utilities \t\t Neighborhood \t\t BldgType \t\t HouseStyle \t\t OverallQual \t\t OverallCond \t\t YearBuilt")
        mytext.insert(END," ------------ \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \n")
        for row in mytext1:
            mytext.insert(END,"       {0} \t\t     {1} \t\t     {2} \t\t     {3} \t\t     {4} \t\t     {5} \t\t     {6} \t\t     {7} \t\t     {8} \t\t     {9}\n".format(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]))
        mytext.pack( side = LEFT)
    
    
             
def searchData():
        e6 = entry1.get()
        mycursor.execute("select SalePrice from submission where id = '%s'"%e6)
        mytext2 = mycursor.fetchall()
        entry2.insert(END,mytext2[0])
        
        
def Clear():
        entry1.delete(0, END)
        entry2.delete(0, END)
    
def filterData():
        root = Tk()
        root.configure(bg="Grey")
        root.title("Housing Database")
        def filterBldgType():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            i1 = entry3.get()
            mycursor.execute("select Id, Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt from test where BldgType = '%s'"%i1)
            mytext3 = mycursor.fetchall()
            mytext = Text(root,width=500,height= 200 ,bg= "gray",fg="black", font=("Times", 12))
            mytext.insert(END," Id \t\t Street \t\t Alley \t\t Utilities \t\t Neighborhood \t\t BldgType \t\t HouseStyle \t\t OverallQual \t\t OverallCond \t\t YearBuilt")
            mytext.insert(END," ------------ \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \n")
            for row in mytext3:
                mytext.insert(END,"       {0} \t\t     {1} \t\t     {2} \t\t     {3} \t\t     {4} \t\t     {5} \t\t     {6} \t\t     {7} \t\t     {8} \t\t     {9}\n".format(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]))
            mytext.pack( side = LEFT)
        
        def filterNeighborhood():
            root = Tk()
            root.configure(bg="Grey")
            root.title("Housing Database")
            i1 = entry3.get()
            mycursor.execute("select Id, Street, Alley, Utilities, Neighborhood, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt from test where Neighborhood = '%s'"%i1)
            mytext4 = mycursor.fetchall()
            mytext = Text(root,width=500,height= 200 ,bg= "gray",fg="black", font=("Times", 12))
            mytext.insert(END," Id \t\t Street \t\t Alley \t\t Utilities \t\t Neighborhood \t\t BldgType \t\t HouseStyle \t\t OverallQual \t\t OverallCond \t\t YearBuilt")
            mytext.insert(END," ------------ \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \t\t---------- \n")
            for row in mytext4:
                mytext.insert(END,"       {0} \t\t     {1} \t\t     {2} \t\t     {3} \t\t     {4} \t\t     {5} \t\t     {6} \t\t     {7} \t\t     {8} \t\t     {9}\n".format(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]))
            mytext.pack( side = LEFT)
        
        entry3= Entry(root, font=("Times", 14),bd=8,width=25,bg="white")
        button4= Button(root,activebackground="green", text="Filter by Building Type",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterBldgType)
        button5= Button(root,activebackground="green", text="Filter by Neighborhood",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterNeighborhood)
        entry3.grid(row=3,column=4, padx=10, pady=10)
        button4.grid(row=4,column=4, padx=10, pady=10)
        button5.grid(row=5,column=4, padx=10, pady=10)
        

buttoncolor="black"
buttonfg="white"
button1= Button(root,activebackground="green", text="SEARCH Predited price",bd=8, bg=buttoncolor, fg=buttonfg, width=25, font=("Times", 12),command=searchData)
button2= Button(root,activebackground="green", text="CLEAR",bd=8, bg=buttoncolor, fg=buttonfg, width =15, font=("Times", 12),command=Clear)
button3= Button(root,activebackground="green", text="VIEW DATABASE",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=showdatabase)
button5= Button(root,activebackground="green", text="FILTER",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=filterData)
button6= Button(root,activebackground="green", text="SHOW ANALYSIS",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=displayGraph)
button7= Button(root,activebackground="green", text="EXIT",bd=8, bg=buttoncolor, fg=buttonfg, width =25, font=("Times", 12),command=qExit)
  
    
#label0.grid(row=1,column=4, padx=10, pady=10)    
label1.grid(row=6,column=0, padx=10, pady=10)
label2.grid(row=7,column=0, padx=10, pady=10)
    
entry1.grid(row=6,column=1, padx=10, pady=10)
entry2.grid(row=7,column=1, padx=10, pady=10)
    
button1.grid(row=8,column=1, padx=10, pady=10)
button2.grid(row=9,column=1, padx=10, pady=10)
button3.grid(row=6,column=6, padx=10, pady=10)
button5.grid(row=7,column=6, padx=10, pady=10)
button6.grid(row=8,column=6, padx=10, pady=10)
button7.grid(row=10,column=6, padx=10, pady=10)


root.mainloop()     
