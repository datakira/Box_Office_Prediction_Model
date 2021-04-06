```python
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import sklearn.metrics as mt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import warnings 
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.7f}'.format

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/lightgbm/__init__.py:48: UserWarning: Starting from version 2.2.1, the library file in distribution wheels for macOS is built by the Apple Clang (Xcode_8.3.3) compiler.
    This means that in case of installing LightGBM from PyPI via the ``pip install lightgbm`` command, you don't need to install the gcc compiler anymore.
    Instead of that, you need to install the OpenMP library, which is required for running LightGBM on the system with the Apple Clang compiler.
    You can install the OpenMP library by the following command: ``brew install libomp``.
      "You can install the OpenMP library by the following command: ``brew install libomp``.", UserWarning)
    


```python
import xgboost as xgb
```


```python
xgb.__version__
```




    '0.90'




```python

```


```python
temp = pd.read_excel("data/최종낫널.xlsx")
```


```python
temp.mpaa = temp.mpaa.apply(lambda x:1 if x == 0 else ( 2 if x == 7 
                        else(3 if x ==12 or x==13 else (4 if x == 15 
                        else 5))))
```


```python
temp.drop(['kor_revenue'], axis = 1, inplace = True)
```


```python
temp['kor_audience_class'] = temp.kor_audience.apply(lambda x: "A" if x >= 5000000 else("B" if x >= 1000000 else( "C" if x >= 100000 else ( "D" if x >= 1000 else "F"))))
```


```python
full_cols = ['movienm', 'showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance', 'horror', 'etc',
       'documentary', 'drama', 'romance', 'musical', 'mystery', 'crime',
       'history', 'western', 'adult', 'thriller', 'animation', 'action',
       'adventure', 'war', 'comedy', 'fantasy', 'director_appearance',
       'director_revenue', 'distributor_share', 'opendt_quarter', 'year_gap',
       'showtypes_num', 'actor_score', 'neg', 'pos', 'neu', 'compound',
       'story_com', 'music_com', 'direction_com', 'actor_com', 'acting_com',
       'story_pos', 'music_pos', 'direction_pos', 'actor_pos', 'acting_pos',
       'story_neg', 'music_neg', 'direction_neg', 'actor_neg', 'acting_neg',
       'x0', 'x1', 'x2', 'x3', 'x4', 'label', 'center_angle1', 'center_angle2',
       'center_angle3', 'center_angle4', 'center_angle5', 'center_angle6',
       'center_angle7', 'center_angle8', 'center_angle9', 'center_angle10']

x_cols1 =['showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance', 'horror', 'etc',
       'documentary', 'drama', 'romance', 'musical', 'mystery', 'crime',
       'history', 'western', 'adult', 'thriller', 'animation', 'action',
       'adventure', 'war', 'comedy', 'fantasy', 'director_appearance',
        'distributor_share',
       'showtypes_num', 'actor_score','compound',
       'story_com', 'music_com', 'direction_com', 'actor_com', 'acting_com',
       'x0', 'x1', 'x2', 'x3', 'x4']

x_cols2 = ['showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance', 'horror', 'etc',
       'documentary', 'drama', 'romance', 'musical', 'mystery', 'crime',
       'history', 'western', 'adult', 'thriller', 'animation', 'action',
       'adventure', 'war', 'comedy', 'fantasy', 'director_appearance',
        'distributor_share',
       'showtypes_num', 'actor_score', 'x0', 'x1', 'x2', 'x3', 'x4'
]

x_cols3 = ['showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance', 'horror', 'etc',
       'documentary', 'drama', 'romance', 'musical', 'mystery', 'crime',
       'history', 'western', 'adult', 'thriller', 'animation', 'action',
       'adventure', 'war', 'comedy', 'fantasy', 'director_appearance',
        'distributor_share',
       'showtypes_num', 'actor_score']

y_cols = ['kor_audience_class']


X = temp[x_cols1]
X2 = temp[x_cols2]
X3 = temp[x_cols3]
Y = temp[y_cols]
```


```python
transformer = StandardScaler().fit(X)
X = transformer.transform(X)

transformer = StandardScaler().fit(X2)
X2 = transformer.transform(X2)

transformer = StandardScaler().fit(X3)
X3 = transformer.transform(X3)
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X, Y,\
                    test_size = 0.3, \
                    random_state = 55)
X_train2, X_test2, y_train2, y_test2 =\
train_test_split(X2, Y,\
                    test_size = 0.3, \
                    random_state = 55)
X_train3, X_test3, y_train3, y_test3 =\
train_test_split(X3, Y,\
                    test_size = 0.3, \
                    random_state = 55)

```


```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 0)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)
X_train_over2, y_train_over2 = smote.fit_sample(X_train2, y_train2)
X_train_over3, y_train_over3 = smote.fit_sample(X_train3, y_train3)
```


```python
xgb_clf = XGBClassifier(n_estimators = 100, max_depth = 3, max_features =5)
xgb_clf.fit(X_train_over, y_train_over, early_stopping_rounds = 100, 
             eval_set = [(X_test, y_test)], eval_metric = 'merror',verbose = False)
#평가
print('학습데이터 정확도:', xgb_clf.score(X_train_over, y_train_over))
print('테스트데이터 정확도:', xgb_clf.score(X_test, y_test))
```

    학습데이터 정확도: 0.8501915708812261
    테스트데이터 정확도: 0.723018147086915
    


```python
xgb_clf2 = XGBClassifier(n_estimators = 100, max_depth = 3, max_features =5)
xgb_clf2.fit(X_train_over2, y_train_over2, early_stopping_rounds = 100, 
             eval_set = [(X_test2, y_test2)], eval_metric = 'merror',verbose = False)
#평가
print('학습데이터 정확도:', xgb_clf2.score(X_train_over2, y_train_over2))
print('테스트데이터 정확도:', xgb_clf2.score(X_test2, y_test2))
```

    학습데이터 정확도: 0.8507662835249042
    테스트데이터 정확도: 0.708691499522445
    


```python
xgb_clf3 = XGBClassifier(n_estimators = 100, max_depth = 3, max_features =5)
xgb_clf3.fit(X_train_over3, y_train_over3, early_stopping_rounds = 100, 
             eval_set = [(X_test3, y_test3)], eval_metric = 'merror',verbose = False)
#평가
print('학습데이터 정확도:', xgb_clf3.score(X_train_over3, y_train_over3))
print('테스트데이터 정확도:', xgb_clf3.score(X_test3, y_test3))
```

    학습데이터 정확도: 0.8310344827586207
    테스트데이터 정확도: 0.6972301814708691
    


```python
y_pred = xgb_clf.predict(X_test)
y_pred2 = xgb_clf2.predict(X_test2)
y_pred3 = xgb_clf3.predict(X_test3)
```


```python
def get_clf_eval(y_test,pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')
    print("오차행렬")
    print(confusion)
    print("정확도",accuracy)
    print("정밀도",precision)
    print("재현율",recall)
    print("F1",f1)
```


```python
get_clf_eval(y_test, y_pred)
```

    오차행렬
    [[  3   2   0   0   0]
     [  8  44  26   5   1]
     [  2  32 120  39   1]
     [  0   2  49 209  67]
     [  0   0   5  51 381]]
    정확도 0.723018147086915
    정밀도 0.5829871794871795
    재현율 0.6506727005276114
    F1 0.6001143722120998
    


```python
get_clf_eval(y_test2, y_pred2)
```

    오차행렬
    [[  1   4   0   0   0]
     [ 11  42  26   4   1]
     [  2  32 126  33   1]
     [  0   3  52 201  71]
     [  0   0   6  59 372]]
    정확도 0.708691499522445
    정밀도 0.5405339645789085
    재현율 0.5630844032801482
    F1 0.5451769254840051
    


```python
get_clf_eval(y_test3, y_pred3)
```

    오차행렬
    [[  1   3   1   0   0]
     [ 10  46  23   4   1]
     [  3  36 115  38   2]
     [  0   2  56 196  73]
     [  0   0   5  60 372]]
    정확도 0.6972301814708691
    정밀도 0.5326478934549983
    재현율 0.5582099026428555
    F1 0.5389818330161059
    


```python
pd.DataFrame(
[[accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='macro'),recall_score(y_test, y_pred, average='macro')],
 [accuracy_score(y_test, y_pred2),precision_score(y_test, y_pred2, average='macro'),recall_score(y_test, y_pred2, average='macro')],
 [accuracy_score(y_test, y_pred3),precision_score(y_test, y_pred3, average='macro'),recall_score(y_test, y_pred3, average='macro')]], index=['R&S', 'S','None'] ,columns=['정확도','정밀도','재현율']).style.background_gradient(cmap='summer')
```




<style  type="text/css" >
    #T_92093f72_f0e4_11ea_a9ed_acde48001122row0_col0 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row0_col1 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row0_col2 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row1_col0 {
            background-color:  #71b866;
            color:  #000000;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row1_col1 {
            background-color:  #289366;
            color:  #000000;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row1_col2 {
            background-color:  #0d8666;
            color:  #000000;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row2_col0 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row2_col1 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_92093f72_f0e4_11ea_a9ed_acde48001122row2_col2 {
            background-color:  #008066;
            color:  #f1f1f1;
        }</style><table id="T_92093f72_f0e4_11ea_a9ed_acde48001122" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >정확도</th>        <th class="col_heading level0 col1" >정밀도</th>        <th class="col_heading level0 col2" >재현율</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_92093f72_f0e4_11ea_a9ed_acde48001122level0_row0" class="row_heading level0 row0" >R&S</th>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row0_col0" class="data row0 col0" >0.723018</td>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row0_col1" class="data row0 col1" >0.582987</td>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row0_col2" class="data row0 col2" >0.650673</td>
            </tr>
            <tr>
                        <th id="T_92093f72_f0e4_11ea_a9ed_acde48001122level0_row1" class="row_heading level0 row1" >S</th>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row1_col0" class="data row1 col0" >0.708691</td>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row1_col1" class="data row1 col1" >0.540534</td>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row1_col2" class="data row1 col2" >0.563084</td>
            </tr>
            <tr>
                        <th id="T_92093f72_f0e4_11ea_a9ed_acde48001122level0_row2" class="row_heading level0 row2" >None</th>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row2_col0" class="data row2 col0" >0.697230</td>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row2_col1" class="data row2 col1" >0.532648</td>
                        <td id="T_92093f72_f0e4_11ea_a9ed_acde48001122row2_col2" class="data row2 col2" >0.558210</td>
            </tr>
    </tbody></table>




```python

```


```python
from xgboost import plot_importance
```


```python
plot_importance(xgb_clf)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9eef650210>




    
![png](SEULGI_final_modeling_files/SEULGI_final_modeling_23_1.png)
    



```python
plot_importance(xgb_clf2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9eedd46210>




    
![png](SEULGI_final_modeling_files/SEULGI_final_modeling_24_1.png)
    



```python
plot_importance(xgb_clf3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9eee1e4490>




    
![png](SEULGI_final_modeling_files/SEULGI_final_modeling_25_1.png)
    



```python
temp.columns
```




    Index(['index', 'movienm', 'showtm', 'prdtyear', 'Domestic', 'mpaa', 'raters',
           'ratings', 'kor_audience', 'kor_audience_class', 'sf', 'family',
           'performance', 'horror', 'etc', 'documentary', 'drama', 'romance',
           'musical', 'mystery', 'crime', 'history', 'western', 'adult',
           'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
           'fantasy', 'director_appearance', 'director_revenue',
           'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
           'actor_score', 'neg', 'pos', 'neu', 'compound', 'story_com',
           'music_com', 'direction_com', 'actor_com', 'acting_com', 'story_pos',
           'music_pos', 'direction_pos', 'actor_pos', 'acting_pos', 'story_neg',
           'music_neg', 'direction_neg', 'actor_neg', 'acting_neg', 'x0', 'x1',
           'x2', 'x3', 'x4', 'label', 'center_angle1', 'center_angle2',
           'center_angle3', 'center_angle4', 'center_angle5', 'center_angle6',
           'center_angle7', 'center_angle8', 'center_angle9', 'center_angle10',
           'Budget'],
          dtype='object')




```python
review = ['neg', 'pos', 'neu', 'story_pos',
       'music_pos', 'direction_pos', 'actor_pos', 'acting_pos', 'story_neg',
       'music_neg', 'direction_neg', 'actor_neg', 'acting_neg']
```


```python
corr_df = temp.corr()
```


```python
corr_df[review].loc['kor_audience'].sort_values().plot(kind = 'barh', cmap = 'summer')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9eee697990>




    
![png](SEULGI_final_modeling_files/SEULGI_final_modeling_29_1.png)
    



```python
temp2 = temp[temp.kor_audience_class == 'D']
```


```python
corr_df2 = temp2.corr()
```


```python
corr_df2[review].loc['kor_audience'].sort_values().plot(kind = 'barh', cmap = 'summer')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9eedd24850>




    
![png](SEULGI_final_modeling_files/SEULGI_final_modeling_32_1.png)
    



```python
temp3 = temp[temp.kor_audience_class == 'A']
```


```python
corr_df3 = temp3.corr()
```


```python
corr_df3[review].loc['kor_audience'].sort_values().plot(kind = 'barh', cmap = 'summer')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9eee066a90>




    
![png](SEULGI_final_modeling_files/SEULGI_final_modeling_35_1.png)
    



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
