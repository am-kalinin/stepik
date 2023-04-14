import pandas as pd, numpy as np, seaborn as sns
import os
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

plt.rcParams['figure.figsize']=(10, 6)
path = './'

df = pd.read_csv(os.path.join(path, 'diamonds_moded.csv'), sep=';')
print (df.shape)
#input ('---')
print (df.head(5))
#input ('---')
#print (df.isna().sum())
df['color'].fillna(df['color'].mode()[0], inplace=True)
#input ('---')
#print (df.isna().sum())
#input ('---')
cat_columns = [cname for cname in df.columns if df[cname].dtype == 'object']
encoder = preprocessing.LabelEncoder()
print(cat_columns)

for col in cat_columns:
    df[col] = encoder.fit_transform(df[col])

#print(df.head(5))
#print(df.boxplot(figsize=(20,3)))

X = df.drop('price', axis=1)
y = df['price']
X_train, X_valid, y_train, y_valid, = train_test_split(X,y,test_size = 0.2, random_state = 1)

#lr = LinearRegression()
#scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#print ('Linear Regression cross validation MAE:', np.mean(scores))

#dt = DecisionTreeRegressor()
#scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#print ('Decision Tree cross validation:', np.mean(scores))

#rf = RandomForestRegressor(random_state=0)
#scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#print ('Random Forest cross validation MAE:', np.mean(scores))

lgb = lightgbm.LGBMRegressor(random_state=0)
scores = cross_val_score(lgb, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print('Cross validation MAE:', np.mean(scores))

def cv_params(model, param_grid):
    scoring = 'neg_mean_absolute_error'

    opt_params = GridSearchCV (
        estimator = model,
        param_grid = param_grid,
        scoring = scoring,
        cv = 5,
        n_jobs = -1
    )

    opt_params.fit(X_train, y_train)
    params = opt_params.best_params_
    best_score = opt_params.best_score_

    print(f'Best score: {round(-best_score,2)}')
    print(f'Best parametrs: {params}\n')

    return params

lgb_param_grid = {
    'max_depth': [10, 15, -1],
    'num_leaves': [25, 35, 45],
    'n_estimators': [100, 500, 600]
}

lgb_clean = lightgbm.LGBMRegressor(random_state=1)
lgb_params = cv_params(lgb_clean, lgb_param_grid)

#rf_param_grid = {
#    'max_depth': [20,25],
#    'n_estimators': [500,800]
#}
#rf_clean = RandomForestRegressor(random_state=1)
#rf_params = cv_params(rf_clean,rf_param_grid)

lgb = lightgbm.LGBMRegressor(**lgb_params)
lgb.fit(X_train, y_train)

#preds = lgb.perdict(X_valid)
preds = lightgbm.LGBMRegressor(**lgb_params).fit(X_train,y_train).predict(X_valid)

print(f'MAPE: {round(mean_absolute_percentage_error(y_valid,preds) * 100,2)}%')
print(f'MAE: {round(mean_absolute_error(y_valid, preds),2)}')

results = pd.DataFrame({'Model': np.round(preds), 'Actual': y_valid})
results = results.reset_index().drop('index',axis=1)
print(results.head(15))

lgb.fit(X,y)

lightgbm.LGBMRegressor(max_depth=)
