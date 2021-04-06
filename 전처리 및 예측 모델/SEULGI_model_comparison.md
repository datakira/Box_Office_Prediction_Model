```python
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
import sklearn.metrics as mt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import pandas as pd
import warnings 
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/lightgbm/__init__.py:48: UserWarning: Starting from version 2.2.1, the library file in distribution wheels for macOS is built by the Apple Clang (Xcode_8.3.3) compiler.
    This means that in case of installing LightGBM from PyPI via the ``pip install lightgbm`` command, you don't need to install the gcc compiler anymore.
    Instead of that, you need to install the OpenMP library, which is required for running LightGBM on the system with the Apple Clang compiler.
    You can install the OpenMP library by the following command: ``brew install libomp``.
      "You can install the OpenMP library by the following command: ``brew install libomp``.", UserWarning)
    


```python
# temp = pd.read_csv("data/notnulldata.csv")
temp = pd.read_excel('data/최종낫널.xlsx')
temp.drop(['Budget','kor_audience','kor_audience_class'],axis = 1, inplace = True)
```


```python
temp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3490 entries, 0 to 3489
    Data columns (total 72 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   index                3490 non-null   int64  
     1   movienm              3490 non-null   object 
     2   showtm               3490 non-null   int64  
     3   prdtyear             3490 non-null   int64  
     4   Domestic             3490 non-null   int64  
     5   mpaa                 3490 non-null   int64  
     6   raters               3490 non-null   int64  
     7   ratings              3490 non-null   float64
     8   kor_revenue          3490 non-null   int64  
     9   sf                   3490 non-null   int64  
     10  family               3490 non-null   int64  
     11  performance          3490 non-null   int64  
     12  horror               3490 non-null   int64  
     13  etc                  3490 non-null   int64  
     14  documentary          3490 non-null   int64  
     15  drama                3490 non-null   int64  
     16  romance              3490 non-null   int64  
     17  musical              3490 non-null   int64  
     18  mystery              3490 non-null   int64  
     19  crime                3490 non-null   int64  
     20  history              3490 non-null   int64  
     21  western              3490 non-null   int64  
     22  adult                3490 non-null   int64  
     23  thriller             3490 non-null   int64  
     24  animation            3490 non-null   int64  
     25  action               3490 non-null   int64  
     26  adventure            3490 non-null   int64  
     27  war                  3490 non-null   int64  
     28  comedy               3490 non-null   int64  
     29  fantasy              3490 non-null   int64  
     30  director_appearance  3490 non-null   int64  
     31  director_revenue     3490 non-null   float64
     32  distributor_share    3490 non-null   float64
     33  opendt_quarter       3490 non-null   int64  
     34  year_gap             3490 non-null   int64  
     35  showtypes_num        3490 non-null   int64  
     36  actor_score          3490 non-null   float64
     37  neg                  3490 non-null   float64
     38  pos                  3490 non-null   float64
     39  neu                  3490 non-null   float64
     40  compound             3490 non-null   float64
     41  story_com            3490 non-null   float64
     42  music_com            3490 non-null   float64
     43  direction_com        3490 non-null   float64
     44  actor_com            3490 non-null   float64
     45  acting_com           3490 non-null   float64
     46  story_pos            3490 non-null   float64
     47  music_pos            3490 non-null   float64
     48  direction_pos        3490 non-null   float64
     49  actor_pos            3490 non-null   float64
     50  acting_pos           3490 non-null   float64
     51  story_neg            3490 non-null   float64
     52  music_neg            3490 non-null   float64
     53  direction_neg        3490 non-null   float64
     54  actor_neg            3490 non-null   float64
     55  acting_neg           3490 non-null   float64
     56  x0                   3490 non-null   float64
     57  x1                   3490 non-null   float64
     58  x2                   3490 non-null   float64
     59  x3                   3490 non-null   float64
     60  x4                   3490 non-null   float64
     61  label                3490 non-null   int64  
     62  center_angle1        3490 non-null   float64
     63  center_angle2        3490 non-null   float64
     64  center_angle3        3490 non-null   float64
     65  center_angle4        3490 non-null   float64
     66  center_angle5        3490 non-null   float64
     67  center_angle6        3490 non-null   float64
     68  center_angle7        3490 non-null   float64
     69  center_angle8        3490 non-null   float64
     70  center_angle9        3490 non-null   float64
     71  center_angle10       3490 non-null   float64
    dtypes: float64(38), int64(33), object(1)
    memory usage: 1.9+ MB
    


```python
temp.set_index('index', inplace = True)
```


```python
temp.columns
```




    Index(['movienm', 'showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
           'ratings', 'kor_revenue', 'sf', 'family', 'performance', 'horror',
           'etc', 'documentary', 'drama', 'romance', 'musical', 'mystery', 'crime',
           'history', 'western', 'adult', 'thriller', 'animation', 'action',
           'adventure', 'war', 'comedy', 'fantasy', 'director_appearance',
           'director_revenue', 'distributor_share', 'opendt_quarter', 'year_gap',
           'showtypes_num', 'actor_score', 'neg', 'pos', 'neu', 'compound',
           'story_com', 'music_com', 'direction_com', 'actor_com', 'acting_com',
           'story_pos', 'music_pos', 'direction_pos', 'actor_pos', 'acting_pos',
           'story_neg', 'music_neg', 'direction_neg', 'actor_neg', 'acting_neg',
           'x0', 'x1', 'x2', 'x3', 'x4', 'label', 'center_angle1', 'center_angle2',
           'center_angle3', 'center_angle4', 'center_angle5', 'center_angle6',
           'center_angle7', 'center_angle8', 'center_angle9', 'center_angle10'],
          dtype='object')




```python
# domestic log 변환

#label_list = ['domestic']
temp['Domestic'] = temp['Domestic'].apply(lambda x : np.log(1+x))
```


```python
full_cols = ['movienm', 'showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings', 'kor_revenue', 'kor_audience', 'kor_audience_class', 'sf',
       'family', 'performance', 'horror', 'etc', 'documentary', 'drama',
       'romance', 'musical', 'mystery', 'crime', 'history', 'western', 'adult',
       'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
       'fantasy', 'director_appearance', 'director_revenue',
       'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
       'actor_score', 'neg', 'pos', 'neu', 'compound', 'story_com',
       'music_com', 'direction_com', 'actor_com', 'acting_com', 'story_pos',
       'music_pos', 'direction_pos', 'actor_pos', 'acting_pos', 'story_neg',
       'music_neg', 'direction_neg', 'actor_neg', 'acting_neg', 'x0', 'x1',
       'x2', 'x3', 'x4', 'label', 'center_angle1', 'center_angle2',
       'center_angle3', 'center_angle4', 'center_angle5', 'center_angle6',
       'center_angle7', 'center_angle8', 'center_angle9', 'center_angle10']

x_cols1 =['showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings','sf',
       'family', 'performance', 'horror', 'etc', 'documentary', 'drama',
       'romance', 'musical', 'mystery', 'crime', 'history', 'western', 'adult',
       'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
       'fantasy', 'director_appearance', 'director_revenue',
       'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
       'actor_score', 'neg', 'pos', 'neu', 'compound', 'story_com',
       'music_com', 'direction_com', 'actor_com', 'acting_com', 'story_pos',
       'music_pos', 'direction_pos', 'actor_pos', 'acting_pos', 'story_neg',
       'music_neg', 'direction_neg', 'actor_neg', 'acting_neg', 'x0', 'x1',
       'x2', 'x3', 'x4', 'label', 'center_angle1', 'center_angle2',
       'center_angle3', 'center_angle4', 'center_angle5', 'center_angle6',
       'center_angle7', 'center_angle8', 'center_angle9', 'center_angle10']

x_cols2 = ['showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings', 'sf',
       'family', 'performance', 'horror', 'etc', 'documentary', 'drama',
       'romance', 'musical', 'mystery', 'crime', 'history', 'western', 'adult',
       'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
       'fantasy', 'director_appearance', 'director_revenue',
       'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
       'actor_score','x0', 'x1',
       'x2', 'x3', 'x4', 'label', 'center_angle1', 'center_angle2',
       'center_angle3', 'center_angle4', 'center_angle5', 'center_angle6',
       'center_angle7', 'center_angle8', 'center_angle9', 'center_angle10']
y_cols = ['kor_revenue']


X = temp[x_cols1]
X2 = temp[x_cols2]
Y = temp[y_cols]
```


```python
transformer = StandardScaler().fit(X)
X = transformer.transform(X)

transformer =  StandardScaler().fit(X2)
X2 = transformer.transform(X2)
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y, test_size=0.3)
```

# Random Forest


```python
from sklearn.model_selection import RandomizedSearchCV

param_space = {"bootstrap": [True],
        "max_depth": [10],
        #"max_features": ['auto', 'sqrt','log2'],
        #"min_samples_leaf": [2, 3, 4],
        #"min_samples_split": [2, 3, 4, 5],
        "n_estimators": [800, 1000]
}

forest_reg = RandomForestRegressor()
forest_rand_search1 = RandomizedSearchCV(forest_reg, param_space, n_iter=32,
                                        scoring="r2", verbose=False, cv=2,
                                        n_jobs=-1, random_state=42)
forest_reg = RandomForestRegressor()
forest_rand_search2 = RandomizedSearchCV(forest_reg, param_space, n_iter=32,
                                        scoring="r2", verbose=False, cv=2,
                                        n_jobs=-1, random_state=42)

forest_rand_search1.fit(X_train, Y_train)
forest_rand_search2.fit(X_train2,Y_train2)

Y_pred11 = forest_rand_search1.predict(X_test)
Y_pred12 = forest_rand_search2.predict(X_test2)

print(mt.r2_score(Y_test, Y_pred11))
print(mt.r2_score(Y_test2, Y_pred12))
forest_rand_search1.best_score_, forest_rand_search2.best_score_
```

    0.6076799671260702
    0.5855598975823992
    




    (0.416418556593822, 0.3682237833615433)




```python
forest_rand_search1.best_params_
```




    {'n_estimators': 800, 'max_depth': 10, 'bootstrap': True}




```python
forest_rand_search2.best_params_
```




    {'n_estimators': 1000, 'max_depth': 10, 'bootstrap': True}



# XGBoost


```python
from scipy import stats

param_space = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [ 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2]
             }


reg = XGBRegressor(objective='reg:squarederror')
rand_search1 = RandomizedSearchCV(reg, param_space, n_iter=32,
                                        scoring="r2", verbose=False, cv=2,
                                        n_jobs=-1, random_state=42)

reg = XGBRegressor(objective='reg:squarederror')
rand_search2 = RandomizedSearchCV(reg, param_space, n_iter=32,
                                        scoring="r2", verbose=False, cv=2,
                                        n_jobs=-1, random_state=42)

rand_search1.fit(X_train,Y_train)
rand_search2.fit(X_train2,Y_train2)

Y_pred21 = rand_search1.predict(X_test)
Y_pred22 = rand_search2.predict(X_test2)

print(mt.r2_score(Y_test, Y_pred21))
print(mt.r2_score(Y_test2, Y_pred22))

rand_search1.best_score_, rand_search2.best_score_
```

    0.46658814160999673
    0.6124212819468773
    




    (0.4244745899253834, 0.3097760239299573)




```python
rand_search1.best_params_
```




    {'colsample_bytree': 0.8498095607205338,
     'learning_rate': 0.17280941906433755,
     'max_depth': 8,
     'min_child_weight': 2,
     'n_estimators': 366,
     'subsample': 0.5528410587186428}




```python
rand_search2.best_params_
```




    {'colsample_bytree': 0.8634525539522367,
     'learning_rate': 0.04893534826538894,
     'max_depth': 9,
     'min_child_weight': 2,
     'n_estimators': 417,
     'subsample': 0.5266040662428277}




```python

```

# LGBM


```python
param_space = {'max_depth' : [10,20],
            'num_leaves' : [100,200],
            'max_bin':[100,200,300,400],
            'learning_rate':[0.05,0.02]
             }


reg = lgb.LGBMRegressor(max_depth = -1, random_state = 0, learning_state = 0.1, n_estimators = 50 )

lgbm_search1 = RandomizedSearchCV(reg, param_space, n_iter=32,
                                        scoring="r2", verbose=False, cv=2,
                                        n_jobs=-1, random_state=42)

reg = lgb.LGBMRegressor(max_depth = -1, random_state = 0, learning_state = 0.1, n_estimators = 50 )
lgbm_search2 = RandomizedSearchCV(reg, param_space, n_iter=32,
                                        scoring="r2", verbose=False, cv=2,
                                        n_jobs=-1, random_state=42)


lgbm_search1.fit(X_train,Y_train)
lgbm_search2.fit(X_train2,Y_train2)

Y_pred31 = lgbm_search1.predict(X_test)
Y_pred32 = lgbm_search2.predict(X_test2)

print(mt.r2_score(Y_test, Y_pred31))
print(mt.r2_score(Y_test2, Y_pred32))
lgbm_search1.best_score_, lgbm_search2.best_score_
```

    0.5459402622471968
    0.5799764470048704
    




    (0.428136544349516, 0.4168399687899727)




```python
lgbm_search1.best_params_
```




    {'num_leaves': 100, 'max_depth': 20, 'max_bin': 200, 'learning_rate': 0.05}




```python
lgbm_search2.best_params_
```




    {'num_leaves': 100, 'max_depth': 10, 'max_bin': 200, 'learning_rate': 0.05}



# Comparision


```python
pd.DataFrame(
[[mt.r2_score(Y_test, Y_pred11), mt.r2_score(Y_test2, Y_pred12)],
[mt.r2_score(Y_test, Y_pred21), mt.r2_score(Y_test2, Y_pred22)],
[mt.r2_score(Y_test, Y_pred31), mt.r2_score(Y_test2, Y_pred32)]], index=['RF', 'XG','LGBM'] ,columns=['X1', 'X2']).style.background_gradient(cmap='summer')
```




<style  type="text/css" >
    #T_dd086254_ef5f_11ea_9017_acde48001122row0_col0 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_dd086254_ef5f_11ea_9017_acde48001122row0_col1 {
            background-color:  #2c9666;
            color:  #000000;
        }    #T_dd086254_ef5f_11ea_9017_acde48001122row1_col0 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_dd086254_ef5f_11ea_9017_acde48001122row1_col1 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_dd086254_ef5f_11ea_9017_acde48001122row2_col0 {
            background-color:  #8fc766;
            color:  #000000;
        }    #T_dd086254_ef5f_11ea_9017_acde48001122row2_col1 {
            background-color:  #008066;
            color:  #f1f1f1;
        }</style><table id="T_dd086254_ef5f_11ea_9017_acde48001122" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >X1</th>        <th class="col_heading level0 col1" >X2</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_dd086254_ef5f_11ea_9017_acde48001122level0_row0" class="row_heading level0 row0" >RF</th>
                        <td id="T_dd086254_ef5f_11ea_9017_acde48001122row0_col0" class="data row0 col0" >0.607680</td>
                        <td id="T_dd086254_ef5f_11ea_9017_acde48001122row0_col1" class="data row0 col1" >0.585560</td>
            </tr>
            <tr>
                        <th id="T_dd086254_ef5f_11ea_9017_acde48001122level0_row1" class="row_heading level0 row1" >XG</th>
                        <td id="T_dd086254_ef5f_11ea_9017_acde48001122row1_col0" class="data row1 col0" >0.466588</td>
                        <td id="T_dd086254_ef5f_11ea_9017_acde48001122row1_col1" class="data row1 col1" >0.612421</td>
            </tr>
            <tr>
                        <th id="T_dd086254_ef5f_11ea_9017_acde48001122level0_row2" class="row_heading level0 row2" >LGBM</th>
                        <td id="T_dd086254_ef5f_11ea_9017_acde48001122row2_col0" class="data row2 col0" >0.545940</td>
                        <td id="T_dd086254_ef5f_11ea_9017_acde48001122row2_col1" class="data row2 col1" >0.579976</td>
            </tr>
    </tbody></table>




```python

```


```python

```


```python

```
