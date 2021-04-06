```python
!pip install konlpy
!pip install googletrans
!gensim
```


```python
from xgboost import XGBClassifier
import xgboost
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb
import sklearn.metrics as mt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import pandas as pd
import warnings 
warnings.filterwarnings('ignore')

# Read Review Data
import pandas as pd
from tqdm import tqdm
import nltk
import re 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import re 
import tqdm.notebook as tq

%matplotlib inline

from sklearn import svm
from keras.utils import get_file
import os
import gensim
import subprocess
import numpy as np
import random
import requests
import pandas as pd
from IPython.core.pylabtools import figsize
import csv
import matplotlib.pyplot as plt

import gensim.downloader as api

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import konlpy
from konlpy.tag import Okt
from konlpy.tag import Kkma
import re

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()

from googletrans import Translator
translator = Translator()

model = api.load('glove-wiki-gigaword-200')

tqdm.pandas(position=0, leave=True)

nltk.download('wordnet')
nltk.download('vader_lexicon')
```

    [==================================================] 100.0% 252.1/252.1MB downloaded
    

    [nltk_data] Downloading package wordnet to /Users/jeong-
    [nltk_data]     gyeonghui/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.
    [nltk_data] Downloading package vader_lexicon to /Users/jeong-
    [nltk_data]     gyeonghui/nltk_data...
    




    True




```python
temp = pd.read_excel("data/최종일까4.xlsx")
im = pd.read_excel("data/1970_2019_imdb2.xlsx")
df = pd.read_excel("data/1970_2019_영화전체.xlsx")
```


```python
# Data
# English Preprocessing ----------------------------------------------------------------
print("Start English Preprocessing...")
def trim_en(text):
    text = str(text)
    #text =text.lower()
    text=re.sub("[(<.*?>)'']"," ",text)
    text=re.sub("(\\W|\\d)"," ",text)
    text=text.strip()
    
    return text

lemm =  WordNetLemmatizer()
sa = SentimentIntensityAnalyzer()

im.리뷰 = im.리뷰.progress_apply(trim_en)
im.리뷰 = im.리뷰.progress_apply(lambda s:" ".join([lemm.lemmatize(w) for w in str(s).split()]) )


# Sentimental Analysis ----------------------------------------------------------------
print("Start  Sentimental Analysis...")
im['neg'] = 0
im['pos'] = 0
im['neu'] = 0
im['compound'] = 0

for i in tq.tqdm(range(len(im))):
    scores = sa.polarity_scores(str(im.loc[i, '리뷰']))
    im.loc[i,'neg'] = scores['neg']
    im.loc[i,'pos'] = scores['pos']
    im.loc[i,'neu'] = scores['neu']
    im.loc[i,'compound'] = scores['compound']

# Comment Analysis ----------------------------------------------------------------
print("Start  Comment Analysis...")
words = ["스토리", "음악", "연출", "배우","연기"]
translator = Translator()
trans = translator.translate(words, dest="en")

lst2 = [model.most_similar(positive=[lemm.lemmatize(word.text.lower())],topn=50) for word in trans ]
lst2 = [ [trans[k].text] + [j[0] for j in i]  for k,i in enumerate(lst2)]

values = []
values_pos = []
values_neg = []

for i in tq.tqdm(range(len(im))):
    review1 = im.loc[i, '리뷰']
    value = [0.001 for i in range(len(lst2))]
    value_pos = [0.001 for i in range(len(lst2))]
    value_neg = [0.001 for i in range(len(lst2))]
    string_split = str(review1).split()
    for k2, i in enumerate(string_split):
        for k,j in enumerate(lst2):
            if i in j:
                value[k] +=1
                left = max([0, k2-10])
                right = k2+10
                window = " ".join(string_split[left:right])
                scores = sa.polarity_scores(window)
                if scores['compound']>0:
                    value_pos[k] += scores['pos']
                if scores['compound']<0:
                    value_neg[k] += scores['neg']

    #Normalize  
    #v = [value_pos[i]/value[i] for i in range(len(value_pos))]
    #values_pos.append(v)
    # else
    values_pos.append(value_pos)


    #Normalize
    #v = [value_neg[i]/value[i] for i in range(len(value_neg))]
    #values_neg.append(v)
    values_neg.append(value_neg)

    #Normalize
    #SUM = sum(value)
    #v = [i/SUM for i in value]
    #values.append(v)
    values.append(value)
    
labels = words
temp1 = pd.DataFrame(values, columns = [i+"_com" for i in  labels])
temp2 = pd.DataFrame(values_pos, columns = [i+"_pos" for i in  labels])
temp3 = pd.DataFrame(values_neg, columns = [i+"_neg" for i in  labels])

data1 = pd.concat([im[['neg', 'pos','neu' ,'compound']], temp1, temp2, temp3],axis=1)
data1.info()
data1.head()
```

    Start English Preprocessing...
    


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    Start  Sentimental Analysis...
    


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    Start  Comment Analysis...
    


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-86bf0db597a2> in <module>
         82     values.append(value)
         83 
    ---> 84 labels = words2
         85 temp1 = pd.DataFrame(values, columns = [i+"_com" for i in  labels])
         86 temp2 = pd.DataFrame(values_pos, columns = [i+"_pos" for i in  labels])
    

    NameError: name 'words2' is not defined



```python
def cosim(x,y):
    denom = norm(x)* norm(y)
    if denom==0:
        return 0
    return np.dot(x,y.T)/(norm(x)* norm(y))

def nuclear_norm(x,y):
    return np.sum(np.abs(x-y))

def spectral_norm(x,y):
    return np.max(np.abs(x-y))

def l2_norm(x,y):
    return np.sqrt(np.sum((x-y)**2))

def rbf_kernel(x,y):
    return np.exp(-1/2*(np.sum((x-y)**2)))

def most_similar(X, x, n, method="cosine"):
    if method=="cosine":
        lst = [(cosim(X[x, :], X[i,:]), i) for i in range(len(X))]
        lst.sort(reverse=True)
    if method=="nuclear":
        lst = [(nuclear_norm(X[x, :], X[i,:]), i) for i in range(len(X))]
        lst.sort(reverse=False)
    if method=="spectral":
        lst = [(spectral_norm(X[x, :], X[i,:]), i) for i in range(len(X))]
        lst.sort(reverse=False)
    if method=="l2":
        lst = [(l2_norm(X[x, :], X[i,:]), i) for i in range(len(X))]
        lst.sort(reverse=False)    
    if method=="rbf":
        lst = [(rbf_kernel(X[x, :], X[i,:]), i) for i in range(len(X))]
        lst.sort(reverse=False)    
    return lst[:n]


# Korean 

okt = Okt()
kkma = Kkma()
def trim_ko(x):
    text=re.sub("[(<.*?>)/'…“”']"," ",x)
    #text=re.sub("(\\W|\\d)"," ",text)
    
    #text = kkma.sentences(text)
    #print(text)
    text = " ".join([w[0] for w in okt.pos(text) if w[1]=="Noun" or w[1]=='Verb' or w[1]=="Adjective"])
    #textx= " ".join([w[0] for w in kkma.pos(text) if w[1]=='NNG' or w[1]=='VA' or w[1]=="MAG"  or w[1]=='MAC'])

    #remove whitespace
    text=text.strip()
    return text


df.스토리_ko.fillna("", inplace=True)
df.스토리_ko = df.스토리_ko.astype(str)
df.genres = df.genres.astype(str)

df.스토리_ko = df.스토리_ko.apply(lambda x:x.replace("\n"," "))
df.스토리_ko = df.스토리_ko.progress_apply(trim_ko)

corpus = df.스토리_ko[(df.스토리_ko.notnull()) & (df.스토리_ko.apply(lambda x: True if len(x)>3 else False))]
corpus = corpus.reset_index()

vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(corpus.스토리_ko.values)
X = X.toarray()

pca = PCA(n_components=5)
pca.fit(X)

new_X = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=0).fit(new_X)

data2 = pd.DataFrame([corpus['index'], new_X[:,0], new_X[:,1], new_X[:,2],new_X[:,3], new_X[:,4], kmeans.labels_]).T
data2.columns=['df_index','x0','x1','x2','x3','x4','label']
data2.df_index = data2.df_index.astype(int)
data2.label = data2.label.astype(int)

print("Kmeans...")
for i in tqdm(range(kmeans.cluster_centers_.shape[0])):
    data2['center_angle'+str(i+1)] = 0
    for j in range(len(data2)):
        data2.loc[j, 'center_angle'+str(i+1)] =  cosim(kmeans.cluster_centers_[i,:], data2.iloc[j, 1:kmeans.cluster_centers_.shape[1]+1].values )

data2 = data2.set_index('df_index')
```


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    Kmeans...
    


    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))


    
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-2aa6e6c6c644> in <module>
         84     data2['center_angle'+str(i+1)] = 0
         85     for j in range(len(data2)):
    ---> 86         data2.loc[j, 'center_angle'+str(i+1)] =  cosim(kmeans.cluster_centers_[i,:], data.iloc[j, 1:kmeans.cluster_centers_.shape[1]+1].values )
         87 
         88 data2 = data2.set_index('df_index')
    

    NameError: name 'data' is not defined



```python
temp = temp[['index', 'movienm', 'showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'kor_revenue', 'kor_audience', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score']]

temp = temp.set_index('index')
temp = temp.join(data1)
temp = temp.join(data2)
temp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movienm</th>
      <th>showtm</th>
      <th>prdtyear</th>
      <th>domestic</th>
      <th>mpaa</th>
      <th>raters</th>
      <th>ratings</th>
      <th>kor_revenue</th>
      <th>kor_audience</th>
      <th>sf</th>
      <th>...</th>
      <th>center_angle1</th>
      <th>center_angle2</th>
      <th>center_angle3</th>
      <th>center_angle4</th>
      <th>center_angle5</th>
      <th>center_angle6</th>
      <th>center_angle7</th>
      <th>center_angle8</th>
      <th>center_angle9</th>
      <th>center_angle10</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>드라이브</td>
      <td>100</td>
      <td>2011</td>
      <td>35061555</td>
      <td>15</td>
      <td>228016</td>
      <td>7.2</td>
      <td>156237980</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>조조 래빗</td>
      <td>108</td>
      <td>2019</td>
      <td>33370906</td>
      <td>12</td>
      <td>250206</td>
      <td>7.9</td>
      <td>965083200</td>
      <td>C</td>
      <td>0</td>
      <td>...</td>
      <td>0.323093</td>
      <td>-0.397164</td>
      <td>-0.744790</td>
      <td>0.569103</td>
      <td>-0.155152</td>
      <td>-0.116992</td>
      <td>0.661112</td>
      <td>0.204118</td>
      <td>-0.224096</td>
      <td>0.072463</td>
    </tr>
    <tr>
      <th>10</th>
      <td>존 윅 - 리로드</td>
      <td>122</td>
      <td>2017</td>
      <td>92029184</td>
      <td>18</td>
      <td>351231</td>
      <td>7.5</td>
      <td>2231583601</td>
      <td>C</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>메멘토</td>
      <td>113</td>
      <td>2000</td>
      <td>25544867</td>
      <td>15</td>
      <td>1096788</td>
      <td>8.4</td>
      <td>325278300</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>0.204231</td>
      <td>0.775064</td>
      <td>0.311798</td>
      <td>-0.040746</td>
      <td>-0.791138</td>
      <td>0.087187</td>
      <td>-0.185423</td>
      <td>-0.102267</td>
      <td>0.199356</td>
      <td>-0.904310</td>
    </tr>
    <tr>
      <th>25</th>
      <td>존 윅: 특별판</td>
      <td>107</td>
      <td>2014</td>
      <td>43037835</td>
      <td>18</td>
      <td>519268</td>
      <td>7.4</td>
      <td>81322120</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>-0.526295</td>
      <td>-0.083300</td>
      <td>-0.435791</td>
      <td>0.897787</td>
      <td>-0.041482</td>
      <td>0.062484</td>
      <td>-0.422683</td>
      <td>-0.503666</td>
      <td>0.109327</td>
      <td>0.159718</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>쿼터매인</td>
      <td>99</td>
      <td>1986</td>
      <td>3751699</td>
      <td>12</td>
      <td>8643</td>
      <td>4.6</td>
      <td>147000</td>
      <td>F</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5783</th>
      <td>크로커다일 던디2</td>
      <td>112</td>
      <td>1988</td>
      <td>109306210</td>
      <td>15</td>
      <td>844</td>
      <td>3.3</td>
      <td>39000</td>
      <td>F</td>
      <td>0</td>
      <td>...</td>
      <td>0.356034</td>
      <td>0.700931</td>
      <td>0.724647</td>
      <td>-0.735532</td>
      <td>-0.386863</td>
      <td>-0.007150</td>
      <td>-0.099774</td>
      <td>0.123897</td>
      <td>0.083190</td>
      <td>-0.655104</td>
    </tr>
    <tr>
      <th>5788</th>
      <td>탱고와 캐쉬</td>
      <td>100</td>
      <td>1989</td>
      <td>63408614</td>
      <td>15</td>
      <td>307534</td>
      <td>7.5</td>
      <td>266000</td>
      <td>F</td>
      <td>0</td>
      <td>...</td>
      <td>0.127171</td>
      <td>0.796803</td>
      <td>0.808207</td>
      <td>-0.543126</td>
      <td>-0.196476</td>
      <td>-0.188776</td>
      <td>-0.566519</td>
      <td>-0.312025</td>
      <td>0.121394</td>
      <td>-0.431946</td>
    </tr>
    <tr>
      <th>5807</th>
      <td>폴링 인 러브</td>
      <td>110</td>
      <td>1984</td>
      <td>11129057</td>
      <td>12</td>
      <td>12334</td>
      <td>6.5</td>
      <td>4186000</td>
      <td>D</td>
      <td>0</td>
      <td>...</td>
      <td>0.266843</td>
      <td>-0.325129</td>
      <td>-0.248123</td>
      <td>-0.284693</td>
      <td>-0.290279</td>
      <td>-0.126371</td>
      <td>-0.351626</td>
      <td>-0.180282</td>
      <td>0.950724</td>
      <td>-0.118663</td>
    </tr>
    <tr>
      <th>5823</th>
      <td>햄버거 힐</td>
      <td>109</td>
      <td>1987</td>
      <td>13839404</td>
      <td>18</td>
      <td>23328</td>
      <td>6.7</td>
      <td>134000</td>
      <td>F</td>
      <td>0</td>
      <td>...</td>
      <td>-0.207928</td>
      <td>-0.156683</td>
      <td>0.512577</td>
      <td>-0.697381</td>
      <td>0.622814</td>
      <td>0.300185</td>
      <td>0.321735</td>
      <td>0.560205</td>
      <td>-0.446290</td>
      <td>0.356748</td>
    </tr>
  </tbody>
</table>
<p>3490 rows × 72 columns</p>
</div>




```python
temp.columns
```




    Index(['movienm', 'showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
           'ratings', 'kor_revenue', 'kor_audience', 'sf', 'family', 'performance',
           'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
           'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
           'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
           'director_appearance', 'director_revenue', 'distributor_share',
           'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
           'pos', 'neu', 'compound', '스토리_com', '음악_com', '연출_com', '배우_com',
           '연기_com', '스토리_pos', '음악_pos', '연출_pos', '배우_pos', '연기_pos', '스토리_neg',
           '음악_neg', '연출_neg', '배우_neg', '연기_neg', 'x0', 'x1', 'x2', 'x3', 'x4',
           'label', 'center_angle1', 'center_angle2', 'center_angle3',
           'center_angle4', 'center_angle5', 'center_angle6', 'center_angle7',
           'center_angle8', 'center_angle9', 'center_angle10'],
          dtype='object')



## 알고리즘 함수 정의


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
```


```python
def DecisionTreeTest(x_train,x_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    prediction=clf.predict(x_test)
    print("DecisionTreeTest 정확도 : %.4f"%accuracy_score(y_test,prediction))


def RandomForestTest(x_train,x_test, y_train, y_test):
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(x_train, y_train)
    pred = rf_clf.predict(x_test)
    accuracy=accuracy_score(y_test,pred)
    print('랜덤포레스트 정확도 : {0:.4f}'.format(accuracy))

def GBMTest(x_train,x_test, y_train, y_test) :
    #GBM모델 셋업
    gb_clf=GradientBoostingClassifier(random_state=0)
    gb_clf.fit(x_train,y_train)
    gb_pred=gb_clf.predict(x_test)
    gb_accuracy = accuracy_score(y_test,gb_pred)
    print('GradientBoostingClassifier 정확도:{0:.4f}'.format(gb_accuracy))
    
def LightGBMTest(x_train,x_test, y_train, y_test):
    lgbm_clf = LGBMClassifier(random_state=0)
    lgbm_clf.fit(x_train, y_train)
    pred = lgbm_clf.predict(x_test)
    accuracy=accuracy_score(y_test,pred)
    print('LGBMClassifier 정확도:{0:.4f}'.format(accuracy))

def MLPClassifierTest(x_train,x_test, y_train, y_test):
    MLP_clf = MLPClassifier()
    MLP_clf.fit(x_train,y_train)
    pred = MLP_clf.predict(x_test)
    accuracy=accuracy_score(y_test,pred)
    print('MLPClassifier 정확도:{0:.4f}'.format(accuracy))
#     print("MLPClassifier 수행시간 : {0:1.4f}초".format(time.time()-start_time))
    
    #http://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220969601609&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
    
def linearRegression(x_train,x_test, y_train, y_test) : 
    LR=LinearRegression()
    LR.fit(x_train,y_train)
    # y_test2 = LR.predict(x_test2)
    score=LR.score(x_test,y_test)
    print("선형회귀 값 : %.6f"%score)
#     print("Misclassification samples : %d" %(y_train2 != y_test2).sum())

def logisticRegression(x_train,x_test, y_train, y_test) : 
    LR=LogisticRegression(C=1000.0,random_state=0)
    LR.fit(x_train,y_train)
    # y_test2 = LR.predict(x_test2)
    score=LR.score(x_test,y_test)
    print("로지스틱 회귀 값 : %.6f"%score)
#     print("Misclassification samples : %d" %(y_train2 != y_test2).sum())
```


```python

```


```python
pd.options.display.float_format = '{:.7f}'.format
#pd.set_option('display.float_format', None)
```


```python
temp.kor_audience.describe()
```


```python
temp.columns
```




    Index(['movienm', 'showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
           'ratings', 'kor_revenue', 'kor_audience', 'sf', 'family', 'performance',
           'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
           'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
           'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
           'director_appearance', 'director_revenue', 'distributor_share',
           'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
           'pos', 'neu', 'compound', '스토리_com', '음악_com', '연출_com', '배우_com',
           '연기_com', '스토리_pos', '음악_pos', '연출_pos', '배우_pos', '연기_pos', '스토리_neg',
           '음악_neg', '연출_neg', '배우_neg', '연기_neg', 'x0', 'x1', 'x2', 'x3', 'x4',
           'label', 'center_angle1', 'center_angle2', 'center_angle3',
           'center_angle4', 'center_angle5', 'center_angle6', 'center_angle7',
           'center_angle8', 'center_angle9', 'center_angle10'],
          dtype='object')




```python
temp.columns = ['movienm', 'showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'kor_revenue', 'kor_audience', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg', 'x0', 'x1', 'x2', 'x3', 'x4',
       'label', 'center_angle1', 'center_angle2', 'center_angle3',
       'center_angle4', 'center_angle5', 'center_angle6', 'center_angle7',
       'center_angle8', 'center_angle9', 'center_angle10']
```


```python
temp = temp.fillna(0)
```


```python
호에 = temp[['actor_score', 'neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg']]
```


```python
temp.to_excel("최종일까_댓글빈도순.xlsx") # 이거 파일이 저장될테니 밑에 분석하실땐 이 파일 불러서 쓰시면 됩니다.
```


```python

```


```python
temp = temp[temp.compound != 0]
```


```python
temp = temp[temp.compound != 1] #이건 제거해야합니다 compound 1인거 살려야합니다
```


```python
호에
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actor_score</th>
      <th>neg</th>
      <th>pos</th>
      <th>neu</th>
      <th>compound</th>
      <th>story_com</th>
      <th>music_com</th>
      <th>direction_com</th>
      <th>actor_com</th>
      <th>acting_com</th>
      <th>story_pos</th>
      <th>music_pos</th>
      <th>direction_pos</th>
      <th>actor_pos</th>
      <th>acting_pos</th>
      <th>story_neg</th>
      <th>music_neg</th>
      <th>direction_neg</th>
      <th>actor_neg</th>
      <th>acting_neg</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>52.00</td>
      <td>0.100</td>
      <td>0.196</td>
      <td>0.704</td>
      <td>0.9999</td>
      <td>281.001</td>
      <td>29.001</td>
      <td>19.001</td>
      <td>158.001</td>
      <td>62.001</td>
      <td>39.562</td>
      <td>4.993</td>
      <td>3.610</td>
      <td>23.860</td>
      <td>6.999</td>
      <td>11.516</td>
      <td>1.132</td>
      <td>0.301</td>
      <td>6.012</td>
      <td>3.492</td>
    </tr>
    <tr>
      <th>7</th>
      <td>92.25</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6.00</td>
      <td>0.142</td>
      <td>0.183</td>
      <td>0.675</td>
      <td>0.9998</td>
      <td>505.001</td>
      <td>12.001</td>
      <td>11.001</td>
      <td>209.001</td>
      <td>67.001</td>
      <td>60.987</td>
      <td>1.521</td>
      <td>2.413</td>
      <td>24.304</td>
      <td>5.455</td>
      <td>39.561</td>
      <td>1.431</td>
      <td>0.211</td>
      <td>18.843</td>
      <td>8.441</td>
    </tr>
    <tr>
      <th>13</th>
      <td>32.00</td>
      <td>0.111</td>
      <td>0.152</td>
      <td>0.737</td>
      <td>0.9996</td>
      <td>315.001</td>
      <td>0.001</td>
      <td>15.001</td>
      <td>113.001</td>
      <td>79.001</td>
      <td>36.683</td>
      <td>0.001</td>
      <td>2.659</td>
      <td>14.499</td>
      <td>8.829</td>
      <td>15.601</td>
      <td>0.001</td>
      <td>0.087</td>
      <td>4.838</td>
      <td>5.128</td>
    </tr>
    <tr>
      <th>25</th>
      <td>50.00</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5780</th>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>5783</th>
      <td>47.50</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>5788</th>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>5807</th>
      <td>184.00</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>5823</th>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
<p>3490 rows × 20 columns</p>
</div>




```python
temp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1236 entries, 2 to 3996
    Data columns (total 72 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   movienm              1236 non-null   object 
     1   showtm               1236 non-null   int64  
     2   prdtyear             1236 non-null   int64  
     3   domestic             1236 non-null   int64  
     4   mpaa                 1236 non-null   int64  
     5   raters               1236 non-null   int64  
     6   ratings              1236 non-null   float64
     7   kor_revenue          1236 non-null   int64  
     8   kor_audience         1236 non-null   object 
     9   sf                   1236 non-null   int64  
     10  family               1236 non-null   int64  
     11  performance          1236 non-null   int64  
     12  horror               1236 non-null   int64  
     13  etc                  1236 non-null   int64  
     14  documentary          1236 non-null   int64  
     15  drama                1236 non-null   int64  
     16  romance              1236 non-null   int64  
     17  musical              1236 non-null   int64  
     18  mystery              1236 non-null   int64  
     19  crime                1236 non-null   int64  
     20  history              1236 non-null   int64  
     21  western              1236 non-null   int64  
     22  adult                1236 non-null   int64  
     23  thriller             1236 non-null   int64  
     24  animation            1236 non-null   int64  
     25  action               1236 non-null   int64  
     26  adventure            1236 non-null   int64  
     27  war                  1236 non-null   int64  
     28  comedy               1236 non-null   int64  
     29  fantasy              1236 non-null   int64  
     30  director_appearance  1236 non-null   int64  
     31  director_revenue     1236 non-null   float64
     32  distributor_share    1236 non-null   float64
     33  opendt_quarter       1236 non-null   int64  
     34  year_gap             1236 non-null   int64  
     35  showtypes_num        1236 non-null   int64  
     36  actor_score          1236 non-null   float64
     37  neg                  1236 non-null   float64
     38  pos                  1236 non-null   float64
     39  neu                  1236 non-null   float64
     40  compound             1236 non-null   float64
     41  story_com            1236 non-null   float64
     42  music_com            1236 non-null   float64
     43  direction_com        1236 non-null   float64
     44  actor_com            1236 non-null   float64
     45  acting_com           1236 non-null   float64
     46  story_pos            1236 non-null   float64
     47  music_pos            1236 non-null   float64
     48  direction_pos        1236 non-null   float64
     49  actor_pos            1236 non-null   float64
     50  acting_pos           1236 non-null   float64
     51  story_neg            1236 non-null   float64
     52  music_neg            1236 non-null   float64
     53  direction_neg        1236 non-null   float64
     54  actor_neg            1236 non-null   float64
     55  acting_neg           1236 non-null   float64
     56  x0                   1236 non-null   float64
     57  x1                   1236 non-null   float64
     58  x2                   1236 non-null   float64
     59  x3                   1236 non-null   float64
     60  x4                   1236 non-null   float64
     61  label                1236 non-null   float64
     62  center_angle1        1236 non-null   float64
     63  center_angle2        1236 non-null   float64
     64  center_angle3        1236 non-null   float64
     65  center_angle4        1236 non-null   float64
     66  center_angle5        1236 non-null   float64
     67  center_angle6        1236 non-null   float64
     68  center_angle7        1236 non-null   float64
     69  center_angle8        1236 non-null   float64
     70  center_angle9        1236 non-null   float64
     71  center_angle10       1236 non-null   float64
    dtypes: float64(39), int64(31), object(2)
    memory usage: 704.9+ KB
    

## 전체 데이터


```python
X_data = temp[['showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg', 'x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.5984
    랜덤포레스트 정확도 : 0.6792
    GradientBoostingClassifier 정확도:0.7035
    MLPClassifier 정확도:0.4879
    로지스틱 회귀 값 : 0.522911
    LGBMClassifier 정확도:0.6954
    

## 스토리만 빼고


```python
X_data = temp[['showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.6361
    랜덤포레스트 정확도 : 0.6792
    GradientBoostingClassifier 정확도:0.7035
    MLPClassifier 정확도:0.5553
    로지스틱 회귀 값 : 0.522911
    LGBMClassifier 정확도:0.6954
    

## 스토리 데이터로만


```python
X_data = temp[['x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.4367
    랜덤포레스트 정확도 : 0.5175
    GradientBoostingClassifier 정확도:0.4933
    MLPClassifier 정확도:0.5256
    로지스틱 회귀 값 : 0.512129
    LGBMClassifier 정확도:0.5148
    

## 리뷰, 스토리 빼고


```python
X_data = temp[['showtm', 'prdtyear', 'domestic', 'mpaa',
       'raters', 'ratings', 'sf', 'family',
       'performance', 'horror', 'etc', 'documentary', 'drama', 'romance',
       'musical', 'mystery', 'crime', 'history', 'western', 'adult',
       'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
       'fantasy', 'director_appearance', 'director_revenue',
       'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
       'actor_score']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.6523
    랜덤포레스트 정확도 : 0.7278
    GradientBoostingClassifier 정확도:0.7062
    MLPClassifier 정확도:0.5472
    로지스틱 회귀 값 : 0.522911
    LGBMClassifier 정확도:0.6954
    

## 리뷰만 빼고


```python
X_data = temp[['showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.6146
    랜덤포레스트 정확도 : 0.7170
    GradientBoostingClassifier 정확도:0.7224
    MLPClassifier 정확도:0.3369
    로지스틱 회귀 값 : 0.522911
    LGBMClassifier 정확도:0.7224
    

## 리뷰 null 제거


```python
temp1 = temp[['showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg', 'x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10', 'kor_audience']]
```


```python
temp1 = temp1[temp1.compound != 0]
```


```python
X_data = temp1.drop(['kor_audience'], axis = 1)
y_data = temp1.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.5876
    랜덤포레스트 정확도 : 0.6981
    GradientBoostingClassifier 정확도:0.7143
    MLPClassifier 정확도:0.4636
    로지스틱 회귀 값 : 0.530997
    LGBMClassifier 정확도:0.6954
    

## 리뷰 null 제거 + 리뷰 특성 빼고


```python
X_data = temp1[['showtm', 'prdtyear', 'domestic', 'mpaa',
       'raters', 'ratings', 'sf', 'family',
       'performance', 'horror', 'etc', 'documentary', 'drama', 'romance',
       'musical', 'mystery', 'crime', 'history', 'western', 'adult',
       'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
       'fantasy', 'director_appearance', 'director_revenue',
       'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
       'actor_score']]
y_data = temp1.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.6712
    랜덤포레스트 정확도 : 0.7278
    GradientBoostingClassifier 정확도:0.7062
    MLPClassifier 정확도:0.3585
    로지스틱 회귀 값 : 0.522911
    LGBMClassifier 정확도:0.6954
    

## 리뷰 null 제거 + 리뷰 특성 포함


```python
X_data = temp1[['showtm', 'prdtyear', 'domestic', 'mpaa',
       'raters', 'ratings', 'sf', 'family',
       'performance', 'horror', 'etc', 'documentary', 'drama', 'romance',
       'musical', 'mystery', 'crime', 'history', 'western', 'adult',
       'thriller', 'animation', 'action', 'adventure', 'war', 'comedy',
       'fantasy', 'director_appearance', 'director_revenue',
       'distributor_share', 'opendt_quarter', 'year_gap', 'showtypes_num',
       'actor_score', 'neg', 'pos', 'neu', 'compound', 'story_com',
       'music_com', 'direction_com', 'actor_com', 'acting_com', 'story_pos',
       'music_pos', 'direction_pos', 'actor_pos', 'acting_pos', 'story_neg',
       'music_neg', 'direction_neg', 'actor_neg', 'acting_neg']]
y_data = temp1.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.6199
    랜덤포레스트 정확도 : 0.6792
    GradientBoostingClassifier 정확도:0.7035
    MLPClassifier 정확도:0.5013
    로지스틱 회귀 값 : 0.522911
    LGBMClassifier 정확도:0.6954
    

## 스토리 + 댓글로만


```python
X_data = temp[['neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg', 'x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.4744
    랜덤포레스트 정확도 : 0.5364
    GradientBoostingClassifier 정확도:0.5391
    MLPClassifier 정확도:0.5256
    로지스틱 회귀 값 : 0.495957
    LGBMClassifier 정확도:0.5337
    

## 스토리로만..


```python
X_data = temp[['x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 78)
```


```python
DecisionTreeTest(X_train, X_test, y_train, y_test)
RandomForestTest(X_train, X_test, y_train, y_test)
GBMTest(X_train, X_test, y_train, y_test)
MLPClassifierTest(X_train, X_test, y_train, y_test)
logisticRegression(X_train, X_test, y_train, y_test)
LightGBMTest(X_train, X_test, y_train, y_test)
```

    DecisionTreeTest 정확도 : 0.4232
    랜덤포레스트 정확도 : 0.5175
    GradientBoostingClassifier 정확도:0.4933
    MLPClassifier 정확도:0.5256
    로지스틱 회귀 값 : 0.512129
    LGBMClassifier 정확도:0.5148
    


```python

```


```python

```


```python

```

# 데이터 뿔려보기


```python
X_data = temp[['showtm', 'prdtyear', 'domestic', 'mpaa', 'raters',
       'ratings', 'sf', 'family', 'performance',
       'horror', 'etc', 'documentary', 'drama', 'romance', 'musical',
       'mystery', 'crime', 'history', 'western', 'adult', 'thriller',
       'animation', 'action', 'adventure', 'war', 'comedy', 'fantasy',
       'director_appearance', 'director_revenue', 'distributor_share',
       'opendt_quarter', 'year_gap', 'showtypes_num', 'actor_score', 'neg',
       'pos', 'neu', 'compound', 'story_com', 'music_com', 'direction_com',
       'actor_com', 'acting_com', 'story_pos', 'music_pos', 'direction_pos',
       'actor_pos', 'acting_pos', 'story_neg', 'music_neg', 'direction_neg',
       'actor_neg', 'acting_neg', 'x0', 'x1', 'x2', 'x3', 'x4', 'label',
       'center_angle1', 'center_angle2', 'center_angle3', 'center_angle4',
       'center_angle5', 'center_angle6', 'center_angle7', 'center_angle8',
       'center_angle9', 'center_angle10']]
y_data = temp.kor_audience
```


```python
X_train, X_test, y_train, y_test =\
train_test_split(X_data, y_data,\
                    test_size = 0.3, \
                    random_state = 20)
```


```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_sample(X_train,y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-131-be71cb168849> in <module>
          1 from imblearn.over_sampling import SMOTE
          2 smote = SMOTE(random_state=0)
    ----> 3 X_train_over, y_train_over = smote.fit_sample(X_train,y_train)
    

    ~/opt/anaconda3/lib/python3.7/site-packages/imblearn/base.py in fit_resample(self, X, y)
         81         )
         82 
    ---> 83         output = self._fit_resample(X, y)
         84 
         85         y_ = (label_binarize(output[1], np.unique(y))
    

    ~/opt/anaconda3/lib/python3.7/site-packages/imblearn/over_sampling/_smote.py in _fit_resample(self, X, y)
        730 
        731             self.nn_k_.fit(X_class)
    --> 732             nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
        733             X_new, y_new = self._make_samples(
        734                 X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
    

    ~/opt/anaconda3/lib/python3.7/site-packages/sklearn/neighbors/_base.py in kneighbors(self, X, n_neighbors, return_distance)
        617                 "Expected n_neighbors <= n_samples, "
        618                 " but n_samples = %d, n_neighbors = %d" %
    --> 619                 (n_samples_fit, n_neighbors)
        620             )
        621 
    

    ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6



```python
DecisionTreeTest(X_train_over, X_test, y_train_over, y_test)
RandomForestTest(X_train_over, X_test, y_train_over, y_test)
GBMTest(X_train_over, X_test, y_train_over, y_test)
MLPClassifierTest(X_train_over, X_test, y_train_over, y_test)
logisticRegression(X_train_over, X_test, y_train_over, y_test)
LightGBMTest(X_train_over, X_test, y_train_over, y_test)
```


```python

```


```python

```
