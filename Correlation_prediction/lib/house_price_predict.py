#!/usr/bin/python

import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import pyplot



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor



train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

print("Basedate--(line,cloum):",train.shape)
train.describe()
#count:每一列非空值的数量
#mean: 每一列的平均值
#std:每一列的


401670
#min：最小值
#25%：25%分位数，排序之后排在25%位置的数
#50%：50%分位数
#75%：75%分位数
#max:最大值
train.drop(['Id'], axis=1, inplace=True)



def show_missing_data(df):
  df_nan = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending = False)
  df_nan_per = df_nan / df.shape[0] * 100

  print(pd.concat([df_nan, df_nan_per], 
                  axis=1, 
                  keys=['nan Amount', 'Percentage']))

train_tmp = train.drop(['SalePrice'], axis=1)
test_ids = test.Id
test = test.drop(['Id'], axis=1)

total = pd.concat([train_tmp, test]).reset_index(drop=True)

print(show_missing_data(total))


fig, axs = plt.subplots(1, 2, figsize=(15,5))

sns.countplot(x=total[pd.notnull(total.PoolQC)].PoolQC, ax=axs[0])
sns.countplot(x=total.PoolArea, ax=axs[1])

plt.suptitle("Pool Quality vs Pool's Area")
axs[0].set_xlabel("Quality")
axs[1].set_xlabel("Area")
#plt.show()
#plt.show(block=False)

total.PoolQC = total.PoolQC.fillna('NA')
train.PoolQC = train.PoolQC.fillna('NA')
test.PoolQC = test.PoolQC.fillna('NA')

sns.countplot(x=total.PoolQC)


def fillNAValues(na_list):
  for elem in na_list:
    total[elem] = total[elem].fillna('NA')
    train[elem] = train[elem].fillna('NA')
    test[elem] = test[elem].fillna('NA')

na_list = ['MiscFeature', 'Alley', 'Fence', 'GarageFinish', 'GarageQual', 
           'GarageCond', 'GarageType', 'BsmtQual', 'BsmtCond', 
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu']

fillNAValues(na_list)



total_lot = (total[(pd.notnull(total.LotFrontage)) 
            & (total.LotFrontage < 200) 
            & (total.LotArea < 100000)]
          [['LotFrontage','LotArea']])

total_lot.plot.scatter(x='LotFrontage', y='LotArea')
#plt.show()
#plt.show(block=False)

regressor = LinearRegression()
regressor.fit(total_lot.LotArea.to_frame(), total_lot.LotFrontage)

lot_nan_total = total[pd.isnull(total.LotFrontage)].LotArea
lot_nan_train = train[pd.isnull(train.LotFrontage)].LotArea
lot_nan_test = test[pd.isnull(test.LotFrontage)].LotArea

lot_pred_total = regressor.predict(lot_nan_total.to_frame())
lot_pred_train = regressor.predict(lot_nan_train.to_frame())
lot_pred_test = regressor.predict(lot_nan_test.to_frame())

total.loc[total.LotFrontage.isnull(), 'LotFrontage'] = lot_pred_total
train.loc[train.LotFrontage.isnull(), 'LotFrontage'] = lot_pred_train
test.loc[test.LotFrontage.isnull(), 'LotFrontage'] = lot_pred_test

total_lot = (total[(pd.notnull(total.LotFrontage)) 
            & (total.LotFrontage < 200) 
            & (total.LotArea < 100000)]
          [['LotFrontage','LotArea']])

total_lot.plot.scatter(x='LotFrontage', y='LotArea')
total.plot.scatter(x='YearBuilt', y='GarageYrBlt')
total.YearBuilt.corr(total.GarageYrBlt)
total[total.YearBuilt==total.GarageYrBlt].count().YearBuilt.astype('float')/total.shape[0]

total.GarageYrBlt = total.GarageYrBlt.fillna(total.YearBuilt)
train.GarageYrBlt = train.GarageYrBlt.fillna(train.YearBuilt)
test.GarageYrBlt = test.GarageYrBlt.fillna(test.YearBuilt)

total.loc[total.GarageYrBlt>2100, 'GarageYrBlt'] = total.YearBuilt
train.loc[train.GarageYrBlt>2100, 'GarageYrBlt'] = train.YearBuilt
test.loc[test.GarageYrBlt>2100, 'GarageYrBlt'] = test.YearBuilt

def replace_with_mode(dfs, cols):
  for df in dfs:
    for col in cols:
      df[col] = df[col].fillna(df[col].mode()[0])

dfs = [total, train, test]
na_values = ['Electrical', 'Functional', 'Utilities', 
                'Exterior2nd', 'Exterior1st', 'KitchenQual',
                'SaleType', 'MSZoning', 'MasVnrType', 'BsmtHalfBath',
                'BsmtFullBath', 'TotalBsmtSF', 'BsmtUnfSF', 
                'BsmtFinSF2', 'BsmtFinSF1']

replace_with_mode(dfs, na_values)

total.loc[total.MasVnrArea.isnull(),['MasVnrArea','MasVnrType']]

total.MasVnrArea = total.MasVnrArea.fillna(0)
train.MasVnrArea = train.MasVnrArea.fillna(0)
test.MasVnrArea = test.MasVnrArea.fillna(0)

print(show_missing_data(total))
total.loc[total.GarageArea.isnull(),['GarageFinish', 'GarageCars', 'GarageArea']]

total.GarageCars = total.GarageCars.fillna(0)
total.GarageArea = total.GarageArea.fillna(0)

train.GarageCars = train.GarageCars.fillna(0)
train.GarageArea = train.GarageArea.fillna(0)

test.GarageCars = test.GarageCars.fillna(0)
test.GarageArea = test.GarageArea.fillna(0)

total.groupby('BsmtHalfBath').BsmtHalfBath.count()

total.BsmtHalfBath = total.BsmtHalfBath.fillna(0)
total.BsmtFullBath = total.BsmtFullBath.fillna(0)

train.BsmtHalfBath = train.BsmtHalfBath.fillna(0)
train.BsmtFullBath = train.BsmtFullBath.fillna(0)

test.BsmtHalfBath = test.BsmtHalfBath.fillna(0)
test.BsmtFullBath = test.BsmtFullBath.fillna(0)

show_missing_data(total)
show_missing_data(train)
show_missing_data(test)
print('Fix all missing_data @@@@@@@@')
print()
train.SalePrice = np.log(train.SalePrice)

final_total = pd.get_dummies(total).reset_index(drop=True)


final_total.shape

y = train.SalePrice
print("len(y):",len(y))
X = final_total.iloc[:len(y),:]

test = final_total.iloc[len(y):,:]

print(X.shape)
print(test.shape)


fig , ax = plt.subplots(figsize = (10, 5))

sns.boxplot(x=X.OverallQual, y=y)
#plt.show()
#plt.show(block=False)
model_reg = LinearRegression()
model_reg.fit(X,y)

accuracies = cross_val_score(estimator=model_reg, X=X, y=y, cv=10)
print(accuracies.mean())

def print_cv_params(selecter_param, selecter_param_str, parameters): 

  grid_search = GridSearchCV(estimator = model_xgb,
                            param_grid = parameters,
                            scoring = 'neg_mean_squared_error',
                            cv = 10,
                            n_jobs = -1)

  grid_result = grid_search.fit(X, y)

  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

  pyplot.errorbar(selecter_param, means, yerr=stds)
  pyplot.title("XGBoost "+ selecter_param_str + " vs Mean Squared Error")
  pyplot.xlabel(selecter_param_str)
  pyplot.ylabel('Mean Squared Error')

model_xgb = XGBRegressor()

print(datetime.datetime.now())
print("Start adapt parameters...")
print()

n_estimators = range(50, 800, 150)
parameters = dict(n_estimators=n_estimators)
print_cv_params(n_estimators, 'n_estimators', parameters)
print(datetime.datetime.now(),"---------------------------------------1")
print()

learning_rate = np.arange(0.0, 0.2, 0.03)
parameters = dict(learning_rate=learning_rate)
print_cv_params(learning_rate, 'learning_rate', parameters)
print(datetime.datetime.now(),"---------------------------------------2")
print()

max_depth = range(0, 7)
parameters = dict(max_depth=max_depth)
print_cv_params(max_depth, 'max_depth', parameters)
print(datetime.datetime.now(),"---------------------------------------3")
print()

min_child_weight = np.arange(0.5, 2., 0.3)
parameters = dict(min_child_weight=min_child_weight)
print_cv_params(min_child_weight, 'min_child_weight', parameters)
print(datetime.datetime.now(),"---------------------------------------4")
print()

gamma = np.arange(.001, .01, .003)
parameters = dict(gamma=gamma)
print_cv_params(gamma, 'gamma', parameters)
print(datetime.datetime.now(),"---------------------------------------5")
print()

subsample = np.arange(0.3, 1., 0.2)
parameters = dict(subsample=subsample)
print_cv_params(subsample, 'subsample', parameters)
print(datetime.datetime.now(),"---------------------------------------6")
print()

colsample_bytree = np.arange(.6, 1, .1)
parameters = dict(colsample_bytree=colsample_bytree)
print_cv_params(colsample_bytree, 'colsample_bytree', parameters)
print(datetime.datetime.now(),"---------------------------------------7")
print()

print("Finish adapt all 7 params...")
print()

parameters = {  
                'colsample_bytree':[.6],
                'subsample':[.9,1],
                'gamma':[.004],
                'min_child_weight':[1.1,1.3],
                'max_depth':[3,6],
                'learning_rate':[.15,.2],
                'n_estimators':[1000],                                                                    
                'reg_alpha':[0.75],
                'reg_lambda':[0.45],
                'seed':[42]
}

grid_search = GridSearchCV(estimator = model_xgb,
                        param_grid = parameters,
                        scoring = 'neg_mean_squared_error',
                        cv = 5,
                        n_jobs = -1)

model_xgb = grid_search.fit(X, y)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

print(datetime.datetime.now())
print("best_score:",best_score)
print("best_parameters:",best_parameters)


accuracies = cross_val_score(estimator=model_xgb, X=X, y=y, cv=10)
print("accuracies.mean:",accuracies.mean())

y_pred = model_xgb.predict(test)
y_pred = np.floor(np.expm1(y_pred))

submission = pd.concat([test_ids, pd.Series(y_pred)], 
                        axis=1,
                        keys=['Id','SalePrice'])
print("Save predict result to sample_submission.csv...")
submission.to_csv('sample_submission.csv', index = False)
print("Saved!")

print("Will show all figures in the end")
time.sleep(5)
plt.show()
