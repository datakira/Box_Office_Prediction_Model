```
import pandas as pd
temp = pd.read_excel("/content/최종일까4.xlsx")
im = pd.read_excel("/content/drive/My Drive/Data/movie/1970_2019_imdb2.xlsx")
df = pd.read_excel("/content/drive/My Drive/Data/1970_2019_영화전체.xlsx")

```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-7-957a3e771e0e> in <module>()
          1 import pandas as pd
    ----> 2 temp = pd.read_excel("/content/최종일까4.xlsx")
          3 im = pd.read_excel("/content/drive/My Drive/Data/movie/1970_2019_imdb2.xlsx")
          4 df = pd.read_excel("/content/drive/My Drive/Data/1970_2019_영화전체.xlsx")
          5 ko_stopwords = pd.read_excel("/content/drive/My Drive/Data/NLP/한국어_stopwords.xlsx").values
    

    /usr/local/lib/python3.6/dist-packages/pandas/io/excel/_base.py in read_excel(io, sheet_name, header, names, index_col, usecols, squeeze, dtype, engine, converters, true_values, false_values, skiprows, nrows, na_values, keep_default_na, verbose, parse_dates, date_parser, thousands, comment, skipfooter, convert_float, mangle_dupe_cols, **kwds)
        302 
        303     if not isinstance(io, ExcelFile):
    --> 304         io = ExcelFile(io, engine=engine)
        305     elif engine and engine != io.engine:
        306         raise ValueError(
    

    /usr/local/lib/python3.6/dist-packages/pandas/io/excel/_base.py in __init__(self, io, engine)
        822         self._io = stringify_path(io)
        823 
    --> 824         self._reader = self._engines[engine](self._io)
        825 
        826     def __fspath__(self):
    

    /usr/local/lib/python3.6/dist-packages/pandas/io/excel/_xlrd.py in __init__(self, filepath_or_buffer)
         19         err_msg = "Install xlrd >= 1.0.0 for Excel support"
         20         import_optional_dependency("xlrd", extra=err_msg)
    ---> 21         super().__init__(filepath_or_buffer)
         22 
         23     @property
    

    /usr/local/lib/python3.6/dist-packages/pandas/io/excel/_base.py in __init__(self, filepath_or_buffer)
        351             self.book = self.load_workbook(filepath_or_buffer)
        352         elif isinstance(filepath_or_buffer, str):
    --> 353             self.book = self.load_workbook(filepath_or_buffer)
        354         elif isinstance(filepath_or_buffer, bytes):
        355             self.book = self.load_workbook(BytesIO(filepath_or_buffer))
    

    /usr/local/lib/python3.6/dist-packages/pandas/io/excel/_xlrd.py in load_workbook(self, filepath_or_buffer)
         34             return open_workbook(file_contents=data)
         35         else:
    ---> 36             return open_workbook(filepath_or_buffer)
         37 
         38     @property
    

    /usr/local/lib/python3.6/dist-packages/xlrd/__init__.py in open_workbook(filename, logfile, verbosity, use_mmap, file_contents, encoding_override, formatting_info, on_demand, ragged_rows)
        114         peek = file_contents[:peeksz]
        115     else:
    --> 116         with open(filename, "rb") as f:
        117             peek = f.read(peeksz)
        118     if peek == b"PK\x03\x04": # a ZIP file
    

    FileNotFoundError: [Errno 2] No such file or directory: '/content/최종일까4.xlsx'


# Word to Score


```
!pip install konlpy
!pip install googletrans
!pip install gensim
```


```
# Read Review Data
from tqdm import tqdm
import nltk
import re 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import re 
import tqdm.notebook as tq

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

tqdm.pandas(position=0, leave=True)
from googletrans import Translator
translator = Translator()

model = api.load('glove-wiki-gigaword-200')

nltk.download('wordnet')
nltk.download('vader_lexicon')
```

    Requirement already satisfied: googletrans in /usr/local/lib/python3.6/dist-packages (3.0.0)
    Requirement already satisfied: httpx==0.13.3 in /usr/local/lib/python3.6/dist-packages (from googletrans) (0.13.3)
    Requirement already satisfied: rfc3986<2,>=1.3 in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (1.4.0)
    Requirement already satisfied: httpcore==0.9.* in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (0.9.1)
    Requirement already satisfied: chardet==3.* in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (3.0.4)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (2020.6.20)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (1.1.0)
    Requirement already satisfied: idna==2.* in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (2.10)
    Requirement already satisfied: hstspreload in /usr/local/lib/python3.6/dist-packages (from httpx==0.13.3->googletrans) (2020.9.2)
    Requirement already satisfied: h2==3.* in /usr/local/lib/python3.6/dist-packages (from httpcore==0.9.*->httpx==0.13.3->googletrans) (3.2.0)
    Requirement already satisfied: h11<0.10,>=0.8 in /usr/local/lib/python3.6/dist-packages (from httpcore==0.9.*->httpx==0.13.3->googletrans) (0.9.0)
    Requirement already satisfied: contextvars>=2.1; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from sniffio->httpx==0.13.3->googletrans) (2.4)
    Requirement already satisfied: hyperframe<6,>=5.2.0 in /usr/local/lib/python3.6/dist-packages (from h2==3.*->httpcore==0.9.*->httpx==0.13.3->googletrans) (5.2.0)
    Requirement already satisfied: hpack<4,>=3.0 in /usr/local/lib/python3.6/dist-packages (from h2==3.*->httpcore==0.9.*->httpx==0.13.3->googletrans) (3.0.0)
    Requirement already satisfied: immutables>=0.9 in /usr/local/lib/python3.6/dist-packages (from contextvars>=2.1; python_version < "3.7"->sniffio->httpx==0.13.3->googletrans) (0.14)
    hello 여보세요
    there 그곳에
    

    /usr/local/lib/python3.6/dist-packages/smart_open/smart_open_lib.py:254: UserWarning: This function is deprecated, use smart_open.open instead. See the migration notes for details: https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst#migrating-to-the-new-open-function
      'See the migration notes for details: %s' % _MIGRATION_NOTES_URL
    

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package vader_lexicon to /root/nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    




    True




```
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
    v = [value_pos[i]/value[i] for i in range(len(value_pos))]
    values_pos.append(v)
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
    

    /usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5825 entries, 0 to 5824
    Data columns (total 19 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   neg       5825 non-null   float64
     1   pos       5825 non-null   float64
     2   neu       5825 non-null   float64
     3   compound  5825 non-null   float64
     4   스토리_com   5825 non-null   float64
     5   음악_com    5825 non-null   float64
     6   연출_com    5825 non-null   float64
     7   배우_com    5825 non-null   float64
     8   연기_com    5825 non-null   float64
     9   스토리_pos   5825 non-null   float64
     10  음악_pos    5825 non-null   float64
     11  연출_pos    5825 non-null   float64
     12  배우_pos    5825 non-null   float64
     13  연기_pos    5825 non-null   float64
     14  스토리_neg   5825 non-null   float64
     15  음악_neg    5825 non-null   float64
     16  연출_neg    5825 non-null   float64
     17  배우_neg    5825 non-null   float64
     18  연기_neg    5825 non-null   float64
    dtypes: float64(19)
    memory usage: 864.8 KB
    




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
      <th>neg</th>
      <th>pos</th>
      <th>neu</th>
      <th>compound</th>
      <th>스토리_com</th>
      <th>음악_com</th>
      <th>연출_com</th>
      <th>배우_com</th>
      <th>연기_com</th>
      <th>스토리_pos</th>
      <th>음악_pos</th>
      <th>연출_pos</th>
      <th>배우_pos</th>
      <th>연기_pos</th>
      <th>스토리_neg</th>
      <th>음악_neg</th>
      <th>연출_neg</th>
      <th>배우_neg</th>
      <th>연기_neg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
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
      <th>1</th>
      <td>0.0</td>
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
      <th>2</th>
      <td>0.1</td>
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
      <th>3</th>
      <td>0.0</td>
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
      <th>4</th>
      <td>0.0</td>
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
  </tbody>
</table>
</div>




```
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


    
    


    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))


    
    


```
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
      <th>family</th>
      <th>performance</th>
      <th>horror</th>
      <th>etc</th>
      <th>documentary</th>
      <th>drama</th>
      <th>romance</th>
      <th>musical</th>
      <th>mystery</th>
      <th>crime</th>
      <th>history</th>
      <th>western</th>
      <th>adult</th>
      <th>thriller</th>
      <th>animation</th>
      <th>action</th>
      <th>adventure</th>
      <th>war</th>
      <th>comedy</th>
      <th>fantasy</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
      <th>opendt_quarter</th>
      <th>year_gap</th>
      <th>showtypes_num</th>
      <th>actor_score</th>
      <th>neg</th>
      <th>pos</th>
      <th>neu</th>
      <th>compound</th>
      <th>스토리_com</th>
      <th>음악_com</th>
      <th>연출_com</th>
      <th>배우_com</th>
      <th>연기_com</th>
      <th>스토리_pos</th>
      <th>음악_pos</th>
      <th>연출_pos</th>
      <th>배우_pos</th>
      <th>연기_pos</th>
      <th>스토리_neg</th>
      <th>음악_neg</th>
      <th>연출_neg</th>
      <th>배우_neg</th>
      <th>연기_neg</th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>label</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.168718e+07</td>
      <td>0.0016</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1.161431e+08</td>
      <td>0.0000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <td>-0.007540</td>
      <td>-0.075842</td>
      <td>0.022933</td>
      <td>-0.044247</td>
      <td>-0.063058</td>
      <td>3.0</td>
      <td>0.038814</td>
      <td>-0.716655</td>
      <td>0.078862</td>
      <td>0.339878</td>
      <td>-0.375964</td>
      <td>0.672698</td>
      <td>-0.067346</td>
      <td>0.647321</td>
      <td>-0.458175</td>
      <td>0.024889</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3.067639e+07</td>
      <td>0.0407</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8.514956e+06</td>
      <td>0.0020</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
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
      <td>0.046876</td>
      <td>0.049038</td>
      <td>0.000488</td>
      <td>-0.058251</td>
      <td>0.002059</td>
      <td>3.0</td>
      <td>0.135686</td>
      <td>0.410639</td>
      <td>-0.063341</td>
      <td>-0.127295</td>
      <td>0.771331</td>
      <td>-0.067157</td>
      <td>-0.813821</td>
      <td>-0.030215</td>
      <td>0.236628</td>
      <td>-0.900306</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.434594e+07</td>
      <td>0.0407</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
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
      <td>-0.010228</td>
      <td>-0.049613</td>
      <td>-0.044464</td>
      <td>-0.046167</td>
      <td>0.042366</td>
      <td>7.0</td>
      <td>-0.046307</td>
      <td>0.082564</td>
      <td>-0.085325</td>
      <td>-0.468822</td>
      <td>-0.140854</td>
      <td>-0.452838</td>
      <td>-0.074363</td>
      <td>0.905896</td>
      <td>-0.665752</td>
      <td>0.224040</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.250566e+06</td>
      <td>0.0000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3.643540e+07</td>
      <td>0.1035</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
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
      <td>0.043570</td>
      <td>0.108399</td>
      <td>0.023611</td>
      <td>-0.001122</td>
      <td>-0.006706</td>
      <td>8.0</td>
      <td>0.117158</td>
      <td>0.292950</td>
      <td>-0.140839</td>
      <td>0.044887</td>
      <td>0.686769</td>
      <td>0.060973</td>
      <td>-0.373589</td>
      <td>-0.739734</td>
      <td>0.741932</td>
      <td>-0.646291</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.1519</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
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
      <td>0.045302</td>
      <td>0.193306</td>
      <td>-0.049250</td>
      <td>-0.001276</td>
      <td>0.041615</td>
      <td>4.0</td>
      <td>0.023809</td>
      <td>0.518710</td>
      <td>-0.346564</td>
      <td>-0.421523</td>
      <td>0.792947</td>
      <td>-0.476607</td>
      <td>-0.240445</td>
      <td>-0.544054</td>
      <td>0.684770</td>
      <td>-0.420068</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3.709686e+06</td>
      <td>0.1035</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
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
      <td>0.123203</td>
      <td>-0.015277</td>
      <td>-0.057614</td>
      <td>0.089394</td>
      <td>0.025088</td>
      <td>0.0</td>
      <td>0.870916</td>
      <td>0.596280</td>
      <td>-0.155582</td>
      <td>-0.252457</td>
      <td>-0.345137</td>
      <td>-0.367567</td>
      <td>-0.327181</td>
      <td>-0.299501</td>
      <td>-0.344827</td>
      <td>-0.103373</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.1035</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
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
      <td>-0.043474</td>
      <td>0.028113</td>
      <td>0.054370</td>
      <td>0.055605</td>
      <td>0.007342</td>
      <td>3.0</td>
      <td>-0.450857</td>
      <td>-0.168854</td>
      <td>0.339655</td>
      <td>0.504731</td>
      <td>-0.154423</td>
      <td>0.275319</td>
      <td>0.617865</td>
      <td>-0.702057</td>
      <td>0.502751</td>
      <td>0.362143</td>
    </tr>
  </tbody>
</table>
<p>3490 rows × 72 columns</p>
</div>




```
temp[temp['movienm']=='겨울왕국']
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
      <th>family</th>
      <th>performance</th>
      <th>horror</th>
      <th>etc</th>
      <th>documentary</th>
      <th>drama</th>
      <th>romance</th>
      <th>musical</th>
      <th>mystery</th>
      <th>crime</th>
      <th>history</th>
      <th>western</th>
      <th>adult</th>
      <th>thriller</th>
      <th>animation</th>
      <th>action</th>
      <th>adventure</th>
      <th>war</th>
      <th>comedy</th>
      <th>fantasy</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
      <th>opendt_quarter</th>
      <th>year_gap</th>
      <th>showtypes_num</th>
      <th>actor_score</th>
      <th>neg</th>
      <th>pos</th>
      <th>neu</th>
      <th>compound</th>
      <th>스토리_com</th>
      <th>음악_com</th>
      <th>연출_com</th>
      <th>배우_com</th>
      <th>연기_com</th>
      <th>스토리_pos</th>
      <th>음악_pos</th>
      <th>연출_pos</th>
      <th>배우_pos</th>
      <th>연기_pos</th>
      <th>스토리_neg</th>
      <th>음악_neg</th>
      <th>연출_neg</th>
      <th>배우_neg</th>
      <th>연기_neg</th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>label</th>
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
      <th>1777</th>
      <td>겨울왕국</td>
      <td>108</td>
      <td>2013</td>
      <td>400738009</td>
      <td>12</td>
      <td>560064</td>
      <td>7.4</td>
      <td>82521307080</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1.335793e+08</td>
      <td>0.1694</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.046</td>
      <td>0.268</td>
      <td>0.686</td>
      <td>1.0</td>
      <td>167.001</td>
      <td>26.001</td>
      <td>7.001</td>
      <td>64.001</td>
      <td>7.001</td>
      <td>37.281</td>
      <td>4.31</td>
      <td>1.25</td>
      <td>13.223</td>
      <td>0.338</td>
      <td>2.219</td>
      <td>0.816</td>
      <td>0.001</td>
      <td>0.977</td>
      <td>0.297</td>
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
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```
temp[temp['movienm']=='어스']
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
      <th>family</th>
      <th>performance</th>
      <th>horror</th>
      <th>etc</th>
      <th>documentary</th>
      <th>drama</th>
      <th>romance</th>
      <th>musical</th>
      <th>mystery</th>
      <th>crime</th>
      <th>history</th>
      <th>western</th>
      <th>adult</th>
      <th>thriller</th>
      <th>animation</th>
      <th>action</th>
      <th>adventure</th>
      <th>war</th>
      <th>comedy</th>
      <th>fantasy</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
      <th>opendt_quarter</th>
      <th>year_gap</th>
      <th>showtypes_num</th>
      <th>actor_score</th>
      <th>neg</th>
      <th>pos</th>
      <th>neu</th>
      <th>compound</th>
      <th>스토리_com</th>
      <th>음악_com</th>
      <th>연출_com</th>
      <th>배우_com</th>
      <th>연기_com</th>
      <th>스토리_pos</th>
      <th>음악_pos</th>
      <th>연출_pos</th>
      <th>배우_pos</th>
      <th>연기_pos</th>
      <th>스토리_neg</th>
      <th>음악_neg</th>
      <th>연출_neg</th>
      <th>배우_neg</th>
      <th>연기_neg</th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>label</th>
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
      <th>412</th>
      <td>어스</td>
      <td>116</td>
      <td>2019</td>
      <td>175084580</td>
      <td>15</td>
      <td>209068</td>
      <td>6.9</td>
      <td>12586771104</td>
      <td>B</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1.170417e+08</td>
      <td>0.1173</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>31.92</td>
      <td>0.135</td>
      <td>0.188</td>
      <td>0.677</td>
      <td>0.9974</td>
      <td>123.001</td>
      <td>8.001</td>
      <td>5.001</td>
      <td>50.001</td>
      <td>17.001</td>
      <td>14.935</td>
      <td>1.298</td>
      <td>0.367</td>
      <td>6.502</td>
      <td>3.016</td>
      <td>10.415</td>
      <td>0.396</td>
      <td>0.997</td>
      <td>5.195</td>
      <td>0.6</td>
      <td>0.002338</td>
      <td>-0.207558</td>
      <td>-0.032139</td>
      <td>-0.234639</td>
      <td>-0.195512</td>
      <td>7.0</td>
      <td>0.078493</td>
      <td>-0.624001</td>
      <td>-0.158586</td>
      <td>0.040723</td>
      <td>-0.08404</td>
      <td>0.463621</td>
      <td>-0.234789</td>
      <td>0.786541</td>
      <td>-0.390929</td>
      <td>-0.127646</td>
    </tr>
  </tbody>
</table>
</div>




```

values = """SF	6.897317	0.056442	0.032214	0.575971	0.335372	0.037166	0.079923	0.469141	0.413771
가족	8.631818	0.047048	0.013229	0.582215	0.357508	0.033949	0.069785	0.477073	0.419193
공연	8.010000	0.288701	0.147626	0.268308	0.295364	0.191173	0.064015	0.364478	0.380334
공포(호러)	5.551064	0.102583	0.056846	0.511650	0.328922	0.044389	0.076611	0.494285	0.384716
다큐멘터리	8.198421	0.151912	0.016448	0.422084	0.409556	0.082581	0.104008	0.336566	0.476845
드라마	8.111493	0.070156	0.027030	0.518252	0.384562	0.050804	0.075190	0.427861	0.446145
멜로/로맨스	7.978472	0.074606	0.014942	0.567023	0.343429	0.047151	0.070064	0.447307	0.435478
뮤지컬	8.154000	0.348072	0.045269	0.362816	0.243842	0.221074	0.094741	0.333895	0.350290
미스터리	6.628621	0.038032	0.038757	0.536515	0.386696	0.036544	0.080871	0.483196	0.399389
범죄	7.117439	0.046644	0.037167	0.567117	0.349072	0.033048	0.075974	0.449683	0.441294
사극	7.976667	0.021509	0.078599	0.533271	0.366621	0.037533	0.047632	0.422369	0.492466
서부극(웨스턴)	6.722500	0.072643	0.067593	0.536934	0.322830	0.026449	0.063326	0.443889	0.466336
스릴러	6.707849	0.062616	0.061160	0.513240	0.362984	0.030545	0.074922	0.472481	0.422051
애니메이션	8.391441	0.079559	0.019899	0.505829	0.394713	0.063208	0.090650	0.463264	0.382878
액션	6.810346	0.055686	0.045305	0.572117	0.326892	0.036780	0.076935	0.474533	0.411752
어드벤처	7.663667	0.052433	0.034778	0.540306	0.372482	0.043553	0.079389	0.469152	0.407905
전쟁	7.756000	0.036516	0.016633	0.615999	0.330852	0.045915	0.082050	0.434713	0.437322
코미디	7.222879	0.067258	0.023799	0.597824	0.311120	0.040729	0.073608	0.454548	0.431116
판타지	7.048485	0.040566	0.024315	0.583863	0.351256	0.036027	0.075312	0.481063	0.407599"""
values = values.split('\n')
values = [i.split('\t') for i  in values]
genres = [i[0] for i in values]
values = [i[1:] for i in values]
df = pd.DataFrame(values, columns="""평점	v_음악_kr	v_연출_kr	v_배우_kr	v_연기_kr	v_음악_en	v_연출_en	v_배우_en	v_연기_en""".split('\t'), index= genres)
df = df.astype(float)
for i in range(4):
    df['diff' +str(i)] = df.iloc[:,i+1] - df.iloc[:,i+5] 
df
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
      <th>평점</th>
      <th>v_음악_kr</th>
      <th>v_연출_kr</th>
      <th>v_배우_kr</th>
      <th>v_연기_kr</th>
      <th>v_음악_en</th>
      <th>v_연출_en</th>
      <th>v_배우_en</th>
      <th>v_연기_en</th>
      <th>diff0</th>
      <th>diff1</th>
      <th>diff2</th>
      <th>diff3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SF</th>
      <td>6.897317</td>
      <td>0.056442</td>
      <td>0.032214</td>
      <td>0.575971</td>
      <td>0.335372</td>
      <td>0.037166</td>
      <td>0.079923</td>
      <td>0.469141</td>
      <td>0.413771</td>
      <td>0.019276</td>
      <td>-0.047709</td>
      <td>0.106830</td>
      <td>-0.078399</td>
    </tr>
    <tr>
      <th>가족</th>
      <td>8.631818</td>
      <td>0.047048</td>
      <td>0.013229</td>
      <td>0.582215</td>
      <td>0.357508</td>
      <td>0.033949</td>
      <td>0.069785</td>
      <td>0.477073</td>
      <td>0.419193</td>
      <td>0.013099</td>
      <td>-0.056556</td>
      <td>0.105142</td>
      <td>-0.061685</td>
    </tr>
    <tr>
      <th>공연</th>
      <td>8.010000</td>
      <td>0.288701</td>
      <td>0.147626</td>
      <td>0.268308</td>
      <td>0.295364</td>
      <td>0.191173</td>
      <td>0.064015</td>
      <td>0.364478</td>
      <td>0.380334</td>
      <td>0.097528</td>
      <td>0.083611</td>
      <td>-0.096170</td>
      <td>-0.084970</td>
    </tr>
    <tr>
      <th>공포(호러)</th>
      <td>5.551064</td>
      <td>0.102583</td>
      <td>0.056846</td>
      <td>0.511650</td>
      <td>0.328922</td>
      <td>0.044389</td>
      <td>0.076611</td>
      <td>0.494285</td>
      <td>0.384716</td>
      <td>0.058194</td>
      <td>-0.019765</td>
      <td>0.017365</td>
      <td>-0.055794</td>
    </tr>
    <tr>
      <th>다큐멘터리</th>
      <td>8.198421</td>
      <td>0.151912</td>
      <td>0.016448</td>
      <td>0.422084</td>
      <td>0.409556</td>
      <td>0.082581</td>
      <td>0.104008</td>
      <td>0.336566</td>
      <td>0.476845</td>
      <td>0.069331</td>
      <td>-0.087560</td>
      <td>0.085518</td>
      <td>-0.067289</td>
    </tr>
    <tr>
      <th>드라마</th>
      <td>8.111493</td>
      <td>0.070156</td>
      <td>0.027030</td>
      <td>0.518252</td>
      <td>0.384562</td>
      <td>0.050804</td>
      <td>0.075190</td>
      <td>0.427861</td>
      <td>0.446145</td>
      <td>0.019352</td>
      <td>-0.048160</td>
      <td>0.090391</td>
      <td>-0.061583</td>
    </tr>
    <tr>
      <th>멜로/로맨스</th>
      <td>7.978472</td>
      <td>0.074606</td>
      <td>0.014942</td>
      <td>0.567023</td>
      <td>0.343429</td>
      <td>0.047151</td>
      <td>0.070064</td>
      <td>0.447307</td>
      <td>0.435478</td>
      <td>0.027455</td>
      <td>-0.055122</td>
      <td>0.119716</td>
      <td>-0.092049</td>
    </tr>
    <tr>
      <th>뮤지컬</th>
      <td>8.154000</td>
      <td>0.348072</td>
      <td>0.045269</td>
      <td>0.362816</td>
      <td>0.243842</td>
      <td>0.221074</td>
      <td>0.094741</td>
      <td>0.333895</td>
      <td>0.350290</td>
      <td>0.126998</td>
      <td>-0.049472</td>
      <td>0.028921</td>
      <td>-0.106448</td>
    </tr>
    <tr>
      <th>미스터리</th>
      <td>6.628621</td>
      <td>0.038032</td>
      <td>0.038757</td>
      <td>0.536515</td>
      <td>0.386696</td>
      <td>0.036544</td>
      <td>0.080871</td>
      <td>0.483196</td>
      <td>0.399389</td>
      <td>0.001488</td>
      <td>-0.042114</td>
      <td>0.053319</td>
      <td>-0.012693</td>
    </tr>
    <tr>
      <th>범죄</th>
      <td>7.117439</td>
      <td>0.046644</td>
      <td>0.037167</td>
      <td>0.567117</td>
      <td>0.349072</td>
      <td>0.033048</td>
      <td>0.075974</td>
      <td>0.449683</td>
      <td>0.441294</td>
      <td>0.013596</td>
      <td>-0.038807</td>
      <td>0.117434</td>
      <td>-0.092222</td>
    </tr>
    <tr>
      <th>사극</th>
      <td>7.976667</td>
      <td>0.021509</td>
      <td>0.078599</td>
      <td>0.533271</td>
      <td>0.366621</td>
      <td>0.037533</td>
      <td>0.047632</td>
      <td>0.422369</td>
      <td>0.492466</td>
      <td>-0.016024</td>
      <td>0.030967</td>
      <td>0.110902</td>
      <td>-0.125845</td>
    </tr>
    <tr>
      <th>서부극(웨스턴)</th>
      <td>6.722500</td>
      <td>0.072643</td>
      <td>0.067593</td>
      <td>0.536934</td>
      <td>0.322830</td>
      <td>0.026449</td>
      <td>0.063326</td>
      <td>0.443889</td>
      <td>0.466336</td>
      <td>0.046194</td>
      <td>0.004267</td>
      <td>0.093045</td>
      <td>-0.143506</td>
    </tr>
    <tr>
      <th>스릴러</th>
      <td>6.707849</td>
      <td>0.062616</td>
      <td>0.061160</td>
      <td>0.513240</td>
      <td>0.362984</td>
      <td>0.030545</td>
      <td>0.074922</td>
      <td>0.472481</td>
      <td>0.422051</td>
      <td>0.032071</td>
      <td>-0.013762</td>
      <td>0.040759</td>
      <td>-0.059067</td>
    </tr>
    <tr>
      <th>애니메이션</th>
      <td>8.391441</td>
      <td>0.079559</td>
      <td>0.019899</td>
      <td>0.505829</td>
      <td>0.394713</td>
      <td>0.063208</td>
      <td>0.090650</td>
      <td>0.463264</td>
      <td>0.382878</td>
      <td>0.016351</td>
      <td>-0.070751</td>
      <td>0.042565</td>
      <td>0.011835</td>
    </tr>
    <tr>
      <th>액션</th>
      <td>6.810346</td>
      <td>0.055686</td>
      <td>0.045305</td>
      <td>0.572117</td>
      <td>0.326892</td>
      <td>0.036780</td>
      <td>0.076935</td>
      <td>0.474533</td>
      <td>0.411752</td>
      <td>0.018906</td>
      <td>-0.031630</td>
      <td>0.097584</td>
      <td>-0.084860</td>
    </tr>
    <tr>
      <th>어드벤처</th>
      <td>7.663667</td>
      <td>0.052433</td>
      <td>0.034778</td>
      <td>0.540306</td>
      <td>0.372482</td>
      <td>0.043553</td>
      <td>0.079389</td>
      <td>0.469152</td>
      <td>0.407905</td>
      <td>0.008880</td>
      <td>-0.044611</td>
      <td>0.071154</td>
      <td>-0.035423</td>
    </tr>
    <tr>
      <th>전쟁</th>
      <td>7.756000</td>
      <td>0.036516</td>
      <td>0.016633</td>
      <td>0.615999</td>
      <td>0.330852</td>
      <td>0.045915</td>
      <td>0.082050</td>
      <td>0.434713</td>
      <td>0.437322</td>
      <td>-0.009399</td>
      <td>-0.065417</td>
      <td>0.181286</td>
      <td>-0.106470</td>
    </tr>
    <tr>
      <th>코미디</th>
      <td>7.222879</td>
      <td>0.067258</td>
      <td>0.023799</td>
      <td>0.597824</td>
      <td>0.311120</td>
      <td>0.040729</td>
      <td>0.073608</td>
      <td>0.454548</td>
      <td>0.431116</td>
      <td>0.026529</td>
      <td>-0.049809</td>
      <td>0.143276</td>
      <td>-0.119996</td>
    </tr>
    <tr>
      <th>판타지</th>
      <td>7.048485</td>
      <td>0.040566</td>
      <td>0.024315</td>
      <td>0.583863</td>
      <td>0.351256</td>
      <td>0.036027</td>
      <td>0.075312</td>
      <td>0.481063</td>
      <td>0.407599</td>
      <td>0.004539</td>
      <td>-0.050997</td>
      <td>0.102800</td>
      <td>-0.056343</td>
    </tr>
  </tbody>
</table>
</div>




```
df.to_excel("장르별감정차이.xlsx", index=False)
```
