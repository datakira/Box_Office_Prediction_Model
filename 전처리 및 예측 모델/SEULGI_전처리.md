```python
import pandas as pd
import numpy as np
```


```python
data = pd.read_excel('data/1970_2020_raw영화데이터_kobiz_update.xlsx')
```

# 0. 개봉영화만 / 필요없는 컬럼지우기


```python
#개봉상태인 행만
data = data[data.prdtStatNm == '개봉']
```


```python
#prdtYear 널값 제외
data = data[data.prdtYear.notnull()]
```


```python
#showTm 널값 제외
data = data[data.showTm.notnull()]
```


```python
#genres 널값제외
data = data[data.genres.notnull()]
```


```python
#showTypes 널값제외
data = data[data.showTypes.notnull()]
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5732 entries, 2 to 5824
    Data columns (total 20 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   movieCd      5732 non-null   object 
     1   movieNm      5732 non-null   object 
     2   movieNmEn    5722 non-null   object 
     3   showTm       5732 non-null   float64
     4   prdtYear     5732 non-null   float64
     5   openDt       5732 non-null   int64  
     6   prdtStatNm   5732 non-null   object 
     7   genres       5732 non-null   object 
     8   directors    5429 non-null   object 
     9   actors       5261 non-null   object 
     10  showTypes    5732 non-null   object 
     11  audits       5616 non-null   object 
     12  Domestic     4164 non-null   object 
     13  Budget       1912 non-null   object 
     14  Distributor  3760 non-null   object 
     15  review       5021 non-null   object 
     16  MPAA         4119 non-null   object 
     17  stats        5468 non-null   float64
     18  raters       5145 non-null   object 
     19  ratings      5145 non-null   float64
    dtypes: float64(4), int64(1), object(15)
    memory usage: 940.4+ KB
    


```python
#domestic 날리기
```


```python
#stats 날리기
```


```python
data.Domestic.value_counts()
```




    -               493
    $587,774          3
    $237,283,207      3
    $538,690          3
    $14,942,422       2
                   ... 
    $18,272,894       1
    $1,109,146        1
    $84,056,472       1
    $40,914,068       1
    $127,807,262      1
    Name: Domestic, Length: 3567, dtype: int64




```python
data = data[data.Domestic.notnull()][data.Domestic != '-']
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    


```python
data.reset_index(inplace= True)
```


```python
data
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>showTypes</th>
      <th>audits</th>
      <th>Domestic</th>
      <th>Budget</th>
      <th>Distributor</th>
      <th>review</th>
      <th>MPAA</th>
      <th>stats</th>
      <th>raters</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>필름,2D</td>
      <td>청소년관람불가,청소년관람불가</td>
      <td>$35,061,555</td>
      <td>$15,000,000</td>
      <td>FilmDistrict</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.449755e+08</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>2D</td>
      <td>12세이상관람가</td>
      <td>$33,370,906</td>
      <td>$14,000,000</td>
      <td>Fox Searchlight Pictures</td>
      <td>This film was exceptional and one of the best ...</td>
      <td>12</td>
      <td>9.650832e+08</td>
      <td>250,206</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>2D</td>
      <td>청소년관람불가</td>
      <td>$291,477</td>
      <td>NaN</td>
      <td>Blue Fox Entertainment</td>
      <td>Strong performances, intimidating villains, cl...</td>
      <td>18</td>
      <td>9.096300e+06</td>
      <td>3,165</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>2D,4D</td>
      <td>청소년관람불가</td>
      <td>$92,029,184</td>
      <td>NaN</td>
      <td>Lionsgate</td>
      <td>In this 2nd installment of John Wick, the styl...</td>
      <td>18</td>
      <td>2.205583e+09</td>
      <td>351,231</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>필름,2D</td>
      <td>15세관람가,15세이상관람가</td>
      <td>$25,544,867</td>
      <td>$9,000,000</td>
      <td>Newmarket Films</td>
      <td>Thank Goodness I didn't read the reviews poste...</td>
      <td>15</td>
      <td>0.000000e+00</td>
      <td>1,096,788</td>
      <td>8.4</td>
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
      <th>3666</th>
      <td>5819</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224</td>
      <td>개봉</td>
      <td>SF</td>
      <td>NaN</td>
      <td>...</td>
      <td>필름</td>
      <td>중학생이상관람가</td>
      <td>$16,295,774</td>
      <td>$37,000,000</td>
      <td>Universal Pictures</td>
      <td>This is an original Marvel film that is actual...</td>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>42,255</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>필름</td>
      <td>중학생이상관람가</td>
      <td>$139,796</td>
      <td>NaN</td>
      <td>Cannon Film Distributors</td>
      <td>Hanna's War is the true story of Hanna Senesh,...</td>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>365</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>NaN</td>
      <td>...</td>
      <td>필름</td>
      <td>고등학생이상관람가</td>
      <td>$4,636,169</td>
      <td>NaN</td>
      <td>Paramount Pictures</td>
      <td>This courtroom thriller was one of the films t...</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>12,879</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td>NaN</td>
      <td>...</td>
      <td>필름</td>
      <td>중학생이상관람가</td>
      <td>$13,839,404</td>
      <td>NaN</td>
      <td>Paramount Pictures</td>
      <td>This is an excellent depiction of the insanity...</td>
      <td>18</td>
      <td>0.000000e+00</td>
      <td>23,328</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>필름</td>
      <td>연소자관람가</td>
      <td>$16,093,651</td>
      <td>NaN</td>
      <td>New Line Cinema</td>
      <td>Mind you, it's not supposed to be, but it is. ...</td>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>5,794</td>
      <td>4.4</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 21 columns</p>
</div>



# 1. 장르 인코딩


```python
!pip install mlxtend
```

    Requirement already satisfied: mlxtend in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (0.17.3)
    Requirement already satisfied: pandas>=0.24.2 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (1.0.1)
    Requirement already satisfied: scikit-learn>=0.20.3 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (0.23.2)
    Requirement already satisfied: matplotlib>=3.0.0 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (3.1.3)
    Requirement already satisfied: numpy>=1.16.2 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (1.19.1)
    Requirement already satisfied: setuptools in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (46.0.0.post20200309)
    Requirement already satisfied: joblib>=0.13.2 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (0.14.1)
    Requirement already satisfied: scipy>=1.2.1 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from mlxtend) (1.4.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from pandas>=0.24.2->mlxtend) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from pandas>=0.24.2->mlxtend) (2019.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from scikit-learn>=0.20.3->mlxtend) (2.1.0)
    Requirement already satisfied: cycler>=0.10 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.0.0->mlxtend) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.0.0->mlxtend) (2.4.6)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from matplotlib>=3.0.0->mlxtend) (1.1.0)
    Requirement already satisfied: six>=1.5 in /Users/injin/opt/anaconda3/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas>=0.24.2->mlxtend) (1.14.0)
    


```python
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import apriori
```


```python
d = data.genres.fillna('기타').str.split(',').to_list()
```


```python
te = TransactionEncoder()
te_ary = te.fit(d).transform(d)
df = pd.DataFrame(te_ary, columns=te.columns_)
```


```python
df = df.astype(int)
```


```python
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
      <th>SF</th>
      <th>가족</th>
      <th>공연</th>
      <th>공포(호러)</th>
      <th>기타</th>
      <th>다큐멘터리</th>
      <th>드라마</th>
      <th>멜로/로맨스</th>
      <th>뮤지컬</th>
      <th>미스터리</th>
      <th>...</th>
      <th>사극</th>
      <th>서부극(웨스턴)</th>
      <th>성인물(에로)</th>
      <th>스릴러</th>
      <th>애니메이션</th>
      <th>액션</th>
      <th>어드벤처</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
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
      <td>...</td>
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
      <th>3666</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3667</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3668</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3669</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3670</th>
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
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>3671 rows × 21 columns</p>
</div>




```python
df.columns
```




    Index(['SF', '가족', '공연', '공포(호러)', '기타', '다큐멘터리', '드라마', '멜로/로맨스', '뮤지컬',
           '미스터리', '범죄', '사극', '서부극(웨스턴)', '성인물(에로)', '스릴러', '애니메이션', '액션', '어드벤처',
           '전쟁', '코미디', '판타지'],
          dtype='object')




```python
data2 = pd.concat([data,df], axis =1)
```


```python
data2
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>사극</th>
      <th>서부극(웨스턴)</th>
      <th>성인물(에로)</th>
      <th>스릴러</th>
      <th>애니메이션</th>
      <th>액션</th>
      <th>어드벤처</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
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
      <th>3666</th>
      <td>5819</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224</td>
      <td>개봉</td>
      <td>SF</td>
      <td>NaN</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3667</th>
      <td>5820</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3668</th>
      <td>5821</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>NaN</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3669</th>
      <td>5823</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td>NaN</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>3670</th>
      <td>5824</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>3671 rows × 42 columns</p>
</div>




```python

```

# 2. 감독 정제


```python
df = pd.DataFrame(columns = ['director_appearance','director_revenue'])
```


```python
data4 = pd.concat([data2,df])
```

# 2-1. 감독 출연 횟수


```python
d_app = data.directors.value_counts()
```


```python
d_app = d_app.to_dict()
```


```python
d_app
```




    {'스티븐 스필버그': 26,
     '클린트 이스트우드': 20,
     '리들리 스콧': 17,
     '론 하워드': 17,
     '로버트 저메키스': 15,
     '올리버 스톤': 13,
     '조엘 슈마허': 13,
     '팀 버튼': 13,
     '토니 스콧': 12,
     '마틴 스코세이지': 12,
     '리차드 도너': 12,
     '스티븐 소더버그': 12,
     '롤랜드 에머리히': 11,
     '크리스 콜럼버스': 11,
     '마이클 베이': 11,
     '안톤 후쿠아': 11,
     '구스 반 산트': 11,
     '롭 라이너': 11,
     '이반 라이트만': 11,
     '프란시스 포드 코폴라': 10,
     '샘 레이미': 10,
     '브렛 래트너': 10,
     '롭 코헨': 10,
     '라세 할스트롬': 10,
     '레니 할린': 10,
     '피터 하이암스': 10,
     '베리 레빈슨': 10,
     '존 터틀타웁': 10,
     '제임스 맨골드': 10,
     'M. 나이트 샤말란': 10,
     '로저 도널드슨': 10,
     '에드워드 즈윅': 10,
     '테일러 핵포드': 9,
     '브라이언 드 팔마': 9,
     '마크 포스터': 9,
     '로버트 로드리게즈': 9,
     '웨스 크레이븐': 9,
     '피터 버그': 8,
     '쿠엔틴 타란티노': 8,
     '게리 마샬': 8,
     '이안': 8,
     '마이크 니콜스': 8,
     '고어 버빈스키': 8,
     '라자 고스넬': 8,
     '배리 소넨필드': 8,
     '제임스 카메론': 8,
     '필립 노이스': 8,
     '아담 쉥크만': 8,
     '스티브 마이너': 8,
     '존 맥티어난': 8,
     '존 파브로': 8,
     '데이빗 핀처': 8,
     '마틴 캠벨': 7,
     '로버트 레드포드': 7,
     '캐서린 비글로우': 7,
     '스파이크 리': 7,
     '사이먼 웨스트': 7,
     '마이클 만': 7,
     '조 존스톤': 7,
     '윌리엄 프리드킨': 7,
     '빌 콘돈': 7,
     '톰 새디악': 7,
     '리 타마호리': 7,
     '피터 위어': 7,
     '대런 아로노프스키': 7,
     '앤디 테넌트': 7,
     '브라이언 싱어': 7,
     '우디 알렌': 7,
     '앤드류 데이비스': 7,
     '숀 레비': 7,
     '볼프강 페터젠': 7,
     '제이 로치': 6,
     '로버트 알트만': 6,
     '잭 스나이더': 6,
     '앤드류 니콜': 6,
     '프란시스 로렌스': 6,
     '커티스 핸슨': 6,
     '테렌스 맬릭': 6,
     '케빈 레이놀즈': 6,
     '데이빗 크로넨버그': 6,
     '월터 힐': 6,
     'J.J. 에이브럼스': 6,
     '오우삼': 6,
     'F. 게리 그레이': 6,
     '애드리안 라인': 6,
     '스티븐 헤렉': 6,
     '조나단 드미': 6,
     '폴 토마스 앤더슨': 6,
     '토드 필립스': 6,
     '데니스 듀간': 6,
     '피터 첼섬': 6,
     '제임스 완': 6,
     '닐 조단': 6,
     '론 언더우드': 6,
     '웨인 왕': 6,
     '스티븐 소머즈': 6,
     '제임스 그레이': 6,
     'D.J. 카루소': 6,
     '도날드 페트리': 6,
     '폴 그린그래스': 6,
     '게리 플레더': 6,
     '제임스 폴리': 6,
     '시드니 폴락': 6,
     '페니 마샬': 6,
     '알란 파커': 5,
     '폴 W.S. 앤더슨': 5,
     '얀 드봉': 5,
     '브라이언 레반트': 5,
     '조 라이트': 5,
     '제임스 왕': 5,
     '마이클 치미노': 5,
     '우디 앨런': 5,
     '피터 시걸': 5,
     '닉 카사베츠': 5,
     '존 싱글톤': 5,
     '조 단테': 5,
     '존 무어': 5,
     '로베르트 슈벤트케': 5,
     '대니 보일': 5,
     '롭 마샬': 5,
     '크리스토퍼 놀란': 5,
     '노만 주이슨': 5,
     '노라 에프론': 5,
     '게리 로스': 5,
     '그레고리 호블릿': 5,
     '폴 페이그': 5,
     '마이크 뉴웰': 5,
     '루이스 만도키': 5,
     '데이빗 예이츠': 5,
     '프랭크 오즈': 5,
     '스티븐 홉킨스': 5,
     '로버트 벤튼': 5,
     '폴 버호벤': 5,
     '길예르모 델 토로': 5,
     '브래드 버드': 5,
     '마이클 카튼-존스': 5,
     '케네스 브래너': 5,
     '릴리 워쇼스키,라나 워쇼스키': 5,
     '바벳 슈로더': 5,
     '러셀 멀케이': 5,
     '마이클 무어': 5,
     '조나단 리브스만': 5,
     '가이 리치': 5,
     '카를로스 살다나': 5,
     '리차드 링클레이터': 5,
     '저스틴 린': 5,
     '마크 웹': 5,
     '롭 민코프': 5,
     '데이빗 O. 러셀': 5,
     '맥지': 5,
     '마이크 피기스': 5,
     '페이튼 리드': 4,
     '안소니 루소,조 루소': 4,
     '밀로스 포만': 4,
     '알렉산더 페인': 4,
     '게리 위닉': 4,
     '존 랜디스': 4,
     '아벨 페라라': 4,
     '스콧 데릭슨': 4,
     '알렉스 프로야스': 4,
     '존 카펜터': 4,
     '드와이트 H. 리틀': 4,
     '조지 밀러': 4,
     '사이먼 윈서': 4,
     '데이빗 토히': 4,
     '폴 슈레이더': 4,
     '베티 토마스': 4,
     '조나단 린': 4,
     '존 슐레진저': 4,
     '아론 노리스': 4,
     '랜달 크레이저': 4,
     '조나단 카프란': 4,
     '스티븐 개건': 4,
     '안드레이 바르코비악': 4,
     '마이클 호프만': 4,
     '데이빗 프랭클': 4,
     '알폰소 쿠아론': 4,
     '데이빗 레이치': 4,
     '프랭크 마샬': 4,
     '마이크 미첼': 4,
     '데이비드 에이어': 4,
     '알란 J. 파큘라': 4,
     '브루스 베레스포드': 4,
     '시드니 J. 퓨리': 4,
     '해롤드 래미스': 4,
     '루이스 리테리어': 4,
     '조엘 코엔,에단 코엔': 4,
     '브래드 앤더슨': 4,
     '조지 루카스': 4,
     '앤드류 플레밍': 4,
     '라이언 존슨': 4,
     '존 추': 4,
     '도미닉 세나': 4,
     '피터 잭슨': 4,
     '존 G. 어빌드센': 4,
     '데이빗 린치': 4,
     '폴 웨이츠': 4,
     '케빈 스미스': 4,
     '존 아미엘': 4,
     '론 쉘톤': 4,
     '로렌스 캐스단': 4,
     '데이비드 R. 엘리스': 4,
     '존 매든': 4,
     '조지 클루니': 4,
     '로저 스포티스우드': 4,
     '피터 예이츠': 4,
     '존 바담': 4,
     '타셈 싱': 4,
     '대니 캐넌': 4,
     '카메론 크로우': 4,
     '더그 라이만': 4,
     '크리스토퍼 맥쿼리': 4,
     '조셉 루벤': 4,
     '짐 자무시': 4,
     '미미 레더': 4,
     '낸시 마이어스': 4,
     '밥 라펠슨': 4,
     '웨스 앤더슨': 4,
     '찰스 샤이어': 4,
     '맷 리브스': 4,
     '로버트 루게틱': 4,
     '바비 패럴리,피터 패럴리': 4,
     '미카엘 하프스트롬': 4,
     '마크 스티븐 존슨': 4,
     '허버트 로스': 4,
     '마이클 앱티드': 4,
     '레스 메이필드': 4,
     '닐 버거': 4,
     '프랭크 다라본트': 4,
     '드니 빌뇌브': 4,
     '베넷 밀러': 3,
     '데이비드 F. 샌드버그': 3,
     '마크 로렌스': 3,
     '브래드 페이튼': 3,
     '가브리엘 무치노': 3,
     '존 파스킨': 3,
     '로우디 헤링턴': 3,
     '조엘 코엔': 3,
     '존 라세터': 3,
     '테리 길리엄': 3,
     '닐 라부티': 3,
     '톰 듀이': 3,
     '롭 레터맨': 3,
     '세스 맥팔레인': 3,
     '어윈 윙클러': 3,
     '조셉 지토': 3,
     '멜 깁슨': 3,
     '마이크 플래너건': 3,
     '울루 그로스바드': 3,
     '메나헴 골란': 3,
     '키넌 아이보리 웨이언스': 3,
     '게리 트러스데일,커크 와이즈': 3,
     '켄 콰피스': 3,
     '스파이크 존즈': 3,
     '스티브 카': 3,
     '존 힐코트': 3,
     '데이빗 고든 그린': 3,
     '존 리 행콕': 3,
     '마크 앳킨스': 3,
     '잘만 킹': 3,
     '안소니 밍겔라': 3,
     '마커스 니스펠': 3,
     '앤드류 스탠튼': 3,
     '보아즈 데이비슨': 3,
     '리차드 벤자민': 3,
     '앤 플레쳐': 3,
     '데이빗 주커': 3,
     '패트릭 루지어': 3,
     '피터 패럴리,바비 패럴리': 3,
     '로드 루리': 3,
     '제리 주커': 3,
     '티무르 베크맘베토브': 3,
     '조지 P. 코스마토스': 3,
     '로저 컴블': 3,
     '알렉스 켄드릭': 3,
     '로만 폴란스키': 3,
     '존 달': 3,
     '조스 웨던': 3,
     '알란 루돌프': 3,
     '케빈 훅스': 3,
     '조셉 코신스키': 3,
     '브라이언 헬겔랜드': 3,
     '시드니 루멧': 3,
     '마크 펠링톤': 3,
     '폴 해기스': 3,
     '마이클 레만': 3,
     '리 워넬': 3,
     '소피아 코폴라': 3,
     '제이 러셀': 3,
     'P.J. 호건': 3,
     '님로드 안탈': 3,
     '리 다니엘스': 3,
     '존 프랑켄하이머': 3,
     '프랑코 제피렐리': 3,
     '조디 포스터': 3,
     '스탠리 큐브릭': 3,
     '데이빗 코엡': 3,
     '벤 스틸러': 3,
     '존 왓츠': 3,
     '노아 바움백': 3,
     '딘 패리소트': 3,
     '폴 맥기건': 3,
     '리차드 라그라브네스': 3,
     '제프 와드로': 3,
     '조나단 모스토우': 3,
     '찰리 채플린': 3,
     '루이스 로사': 3,
     '안드레이 콘찰로프스키': 3,
     '존 애브넛': 3,
     '존 건': 3,
     '대니 드비토': 3,
     '스콧 힉스': 3,
     '존 스톡웰': 3,
     '브래드 실버링': 3,
     '스콧 스피어': 3,
     '젠디 타타코브스키': 3,
     '팀 스토리': 3,
     '필 알덴 로빈슨': 3,
     '벤 영거': 3,
     '안드레스 무시에티': 3,
     '스튜어트 베이어드': 3,
     '캐서린 하드윅': 3,
     '재리 패리스': 3,
     '캐린 쿠사마': 3,
     '데이빗 S. 워드': 3,
     '데이빗 로워리': 3,
     '알렉산더 아야': 3,
     '션 베이커': 3,
     '데이미언 셔젤': 3,
     '제임스 L. 브룩스': 3,
     '쉐인 블랙': 3,
     '존 폴슨': 3,
     '바즈 루어만': 3,
     '샘 멘데스': 3,
     '해롤드 벡커': 3,
     '존 R. 레오네티': 3,
     '어빈 커쉬너': 3,
     '마크 S. 워터스': 3,
     '데니스 호퍼': 3,
     '패트릭 휴즈': 3,
     '하워드 지프': 3,
     '로드리고 가르시아': 3,
     '하워드 더치': 3,
     '스티븐 프리어스': 3,
     '프랭크 코라치': 3,
     '론 클레멘츠,존 머스커': 3,
     '리차드 론크레인': 3,
     'J.C 챈더': 3,
     '미셸 공드리': 3,
     '폴 웨이랜드': 3,
     '웨스 볼': 3,
     '존 길러민': 3,
     '팀 힐': 3,
     '일라이 로스': 3,
     '블레이크 에드워즈': 3,
     '딘 데블로이스': 3,
     'J. 리 톰슨': 3,
     '빌리 크리스탈': 3,
     '네드 벤슨': 3,
     '존 카메론 밋첼': 3,
     '존 부어만': 3,
     '마틴 브레스트': 3,
     '프레드 쉐피시': 3,
     '척 러셀': 3,
     '마크 레빈': 3,
     '브렛 레너드': 3,
     '케빈 리마': 2,
     '브래드 퍼맨': 2,
     '테이트 테일러': 2,
     '리차드 아텐보로': 2,
     '스티븐 자일리언': 2,
     '실베스터 스탤론': 2,
     '던칸 존스': 2,
     '아넌드 터커': 2,
     '마이클 프레스만': 2,
     '샤론 맥과이어': 2,
     '라이언 쿠글러': 2,
     '토드 헤인즈': 2,
     '로버트 루케틱': 2,
     '크리스 벅,제니퍼 리': 2,
     '안젤리나 졸리': 2,
     '다니엘 스탬': 2,
     '자움 콜렛 세라': 2,
     '스콧 칼벳': 2,
     '어니스트 R. 딕커슨': 2,
     '윌리엄 와일러': 2,
     '제임스 보빈': 2,
     '스테판 루조비츠키': 2,
     '우인태': 2,
     '크레이그 R. 백슬리': 2,
     '브루스 A 에반스': 2,
     '마사 쿨리지': 2,
     '마커스 던스탠': 2,
     '존 휴스턴': 2,
     '짐 길레스피': 2,
     '잭 숄더': 2,
     '발타자르 코루마쿠르': 2,
     '데이빗 슬레이드': 2,
     '루크 그린필드': 2,
     '모건 네빌': 2,
     '데릭 시엔프랜스': 2,
     '매튜 본': 2,
     '폴 마이클 글레이저': 2,
     '스테판 홉킨스': 2,
     '팀 밀러': 2,
     '톰 포드': 2,
     '제이미 블랭크스': 2,
     '마크 로마넥': 2,
     '로저 크리스찬': 2,
     '렌 와이즈먼': 2,
     '레베카 밀러': 2,
     '제시 넬슨': 2,
     '롭 좀비': 2,
     '토마스 맥카시': 2,
     '밥 클락': 2,
     '로슨 마샬 터버': 2,
     '존 헤스': 2,
     '마이크 밀스': 2,
     '조쉬 분': 2,
     '스티브 벡': 2,
     '스티븐 쿼일': 2,
     '리 언크리치': 2,
     '케빈 코스트너': 2,
     '데이빗 돕킨': 2,
     '에단 호크': 2,
     '드레이크 도리머스': 2,
     '제프 니콜스': 2,
     '클락 존슨': 2,
     '버나드 로즈': 2,
     '데이빗 맥킨지': 2,
     '휴 닐슨': 2,
     '존 에릭 도들': 2,
     '리치 무어': 2,
     '로코 데빌리어스': 2,
     '스티브 오데커크': 2,
     '브렉 에이즈너': 2,
     '에이미 헥커링': 2,
     '토머스 카터': 2,
     '데오도르 멜피': 2,
     '니콜 홀로프세너': 2,
     '플로리아 시지스몬디': 2,
     '존 A. 데이비스': 2,
     '마이클 알메레이다': 2,
     '토니 골드윈': 2,
     '루퍼트 와이어트': 2,
     '에밀리오 에스테베즈': 2,
     '커트 위머': 2,
     '에밀 아돌리노': 2,
     '스티븐 벨버': 2,
     '데릭 마티니': 2,
     '롤랑 조페': 2,
     '헤롤드 베커': 2,
     '그레고리 다크': 2,
     '벤 루윈': 2,
     '아담 맥케이': 2,
     '빅터 플레밍': 2,
     '제프 머피': 2,
     '이자벨 코이젯트': 2,
     '댄 스캔론': 2,
     '찰스 허먼-움펠드': 2,
     '돈 코스카렐리': 2,
     '데이브 그린': 2,
     '팻 오코너': 2,
     '데이빗 맥낼리': 2,
     '게리 존스': 2,
     '월트 벡커': 2,
     '믹 잭슨': 2,
     '스티븐 세인버그': 2,
     '낸시 사보카': 2,
     '그레고리 플로킨': 2,
     '개빈 후드': 2,
     '니콜라스 하이트너': 2,
     '캣 코이로': 2,
     '리처드 켈리': 2,
     '안소니 히콕스': 2,
     '데이빗 겔브': 2,
     '데이빗 로버트 밋첼': 2,
     '스캇 워프': 2,
     '조던 복트-로버츠': 2,
     '게빈 오코너': 2,
     '콜린 트레보로우': 2,
     '헨리 유스트,아리엘 슐만': 2,
     '줄리언 슈나벨': 2,
     '조쉬 C. 월러': 2,
     '돈 블루스': 2,
     '조 로스': 2,
     '아서 힐': 2,
     '페드 알바레즈': 2,
     '로버트 하몬': 2,
     '그리핀 던': 2,
     '하우메 콜렛 세라': 2,
     '토니 길로이': 2,
     '존 머스커,론 클레멘츠': 2,
     '스콧 쿠퍼': 2,
     '케빈 먼로': 2,
     '바딤 피얼먼': 2,
     '스티븐 달드리': 2,
     '스티브 바론': 2,
     '바브라 스트라이샌드': 2,
     '웨인 크라머': 2,
     '조쉬 트랭크': 2,
     '닐 블롬캠프': 2,
     '타이카 와이티티': 2,
     '마크 레스터': 2,
     '마이클 위너': 2,
     '기타무라 류헤이': 2,
     '캘리 애스버리': 2,
     '제임스 맥티그': 2,
     '마크 딘달': 2,
     '크리스토퍼 랜던': 2,
     '케빈 브레이': 2,
     '크레이그 질레스피': 2,
     '데이빗 S. 고이어': 2,
     '필립 카우프만': 2,
     '데이빗 호건': 2,
     '진 퀸타노': 2,
     '아담 윈가드': 2,
     '제임스 드모나코': 2,
     '조던 필': 2,
     '마리오 반 피블스': 2,
     '브라이언 핸슨': 2,
     '워렌 비티': 2,
     '마티유 카소비츠': 2,
     '빌리 레이': 2,
     '폴 웨이츠,크리스 웨이츠': 2,
     '글렌 피카라,존 레쿼': 2,
     '케빈 맥도널드': 2,
     '캐롤 발라드': 2,
     '카시 레몬즈': 2,
     '칼 프랭클린': 2,
     '크리스토퍼 케인': 2,
     '조 카나한': 2,
     '앨런 콜터': 2,
     '마크 로코': 2,
     '팀 헌터': 2,
     '마르코 브람빌라': 2,
     '크리스 리노드': 2,
     '피터 패럴리': 2,
     '피터 보그다노비치': 2,
     '알란 메레즈': 2,
     '드와이트 리틀': 2,
     '채드 스타헬스키': 2,
     '스티븐 브릴': 2,
     '벤 애플렉': 2,
     '알버트 파이언': 2,
     '크레이그 조벨': 2,
     '루퍼트 샌더스': 2,
     '롭 바우먼': 2,
     '로렌 스카파리아': 2,
     '마이클 코렌트': 2,
     '앤드류 버그만': 2,
     '그리그 비먼': 2,
     '케니 오테가': 2,
     '크리스 웨이츠': 2,
     '주드 아패토우': 2,
     '울리 에델': 2,
     '제이크 캐스단': 2,
     '아리 애스터': 2,
     '포레스트 휘테커': 2,
     '가스 제닝스': 2,
     '토니 케이': 2,
     '로버트 맨델': 2,
     '앤드류 아담슨': 2,
     '토드 로빈슨': 2,
     '마이클 크리스토퍼': 2,
     '로저 미첼': 2,
     '마크 워터스': 2,
     '데이빗 마멧': 2,
     '칼 쉔켈': 2,
     '앨런 테일러': 2,
     '이안 소프틀리': 2,
     '쉘던 레티치': 2,
     '장 자끄 아노': 2,
     '마샬 허스코비츠': 2,
     '존 터투로': 2,
     '마크 네빌딘,브라이언 테일러': 2,
     '짐 쉐리단': 2,
     '에단 코엔,조엘 코엔': 2,
     '빌 듀크': 2,
     '존 허즈펠드': 2,
     '제임스 건': 2,
     '제레미아 체칙': 2,
     '가렛 에드워즈': 2,
     '다니엘 에스피노사': 2,
     'R. 엘리스 프레이저': 2,
     '그레타 거윅': 2,
     '스티븐 크보스키': 2,
     '랜덜 밀러': 2,
     '제인 캠피온': 2,
     '릭 로먼 워': 2,
     '마이클 오블로위츠': 2,
     '존 글렌': 2,
     '톰 행크스': 2,
     '노암 머로': 2,
     '장 마크 발레': 2,
     '알레한드로 곤잘레스 이냐리투': 2,
     '테드 데미': 2,
     '조 채펠리': 2,
     '라이 루소 영': 2,
     '조 캠프': 2,
     '스티븐 질렌홀': 2,
     '제프 트레마인': 2,
     '마이클 라이머': 2,
     '헨리 셀릭': 2,
     '짐 에이브러햄스': 2,
     '스티븐 노링턴': 2,
     '아리 산델': 2,
     '필 조아누': 2,
     '벤슨 리': 2,
     '디토 몬티엘': 2,
     '윌리엄 유뱅크': 2,
     '키퍼 서덜랜드': 1,
     '호이트 예이트만': 1,
     '프랭크 밀러,쿠엔틴 타란티노': 1,
     '저스틴 프라이스': 1,
     '올리비에 메가턴': 1,
     '한나 피델': 1,
     '빌 우드러프': 1,
     '조나단 헨스라이': 1,
     '로빈 버드,도노반 쿡': 1,
     '바이론 하워드,리치 무어': 1,
     '오토 바서스트': 1,
     '존 카니': 1,
     '마틴 맥도나': 1,
     '존 맥노튼': 1,
     '스튜어트 코넬리': 1,
     '제랄드 코코리티': 1,
     '아리 포신': 1,
     '로드니 에스쳐': 1,
     '리차드 스탠리': 1,
     '브론웬 휴즈': 1,
     '드레이크 도레무스': 1,
     '앨리슨 앤더스,로버트 로드리게즈,쿠엔틴 타란티노,알렉상드르 록웰': 1,
     '오렌 펠리': 1,
     '조나단 밀롯,캐리 멀니온': 1,
     '제시카 골드버그': 1,
     '빈센트 워드': 1,
     '더글라스 맥그래스': 1,
     '폴 마르커스': 1,
     '칼 라이너': 1,
     '줄리 엔템플': 1,
     '니아 발다로스': 1,
     '알레한드로 아메나바르': 1,
     '피터 마누지안': 1,
     '글렌 밀러': 1,
     '모간 스퍼록': 1,
     '크리스 카터': 1,
     '게리 넬슨': 1,
     '후안 안토니오 바요나': 1,
     '플로리안 헨켈 폰 도너스마르크': 1,
     '다니엘 색하임': 1,
     '마이클 브랜트': 1,
     '브루스 맥도널드': 1,
     '찰스 데 라우지리카': 1,
     '이완 맥그리거': 1,
     '다니엘 그로브': 1,
     '캘리 애스버리,앤드류 아담슨,콘래드 베논': 1,
     '스튜어트 고든': 1,
     '윌리엄 H. 머시': 1,
     '로버트 로드리게즈,에단 마니퀴스': 1,
     '알랭 베를리네': 1,
     '대니 스트롱': 1,
     '케이트 베커-플로이랜드': 1,
     '랜드 라비치': 1,
     '로버트 래들러': 1,
     '크리스틴 라티': 1,
     '데미안 해리스': 1,
     '니콜라스 패클러': 1,
     '스티븐 R. 몬로': 1,
     '마이클 윈터바텀': 1,
     '리차드 프랭클린': 1,
     '마이클 크라이튼': 1,
     '존 맥티어난,마이클 크라이튼': 1,
     '존 웰스': 1,
     '허셜 파버': 1,
     '에스펜 샌버그,요아킴 뢰닝': 1,
     '샘 에스마일': 1,
     '버즈 쿨릭': 1,
     '로버트 쿠르츠만': 1,
     '데이비드 레프,존 쉐인필드': 1,
     '테아 샤록': 1,
     '스티븐 힐렌버그': 1,
     '토니 레온디스': 1,
     '롭 보우먼': 1,
     '아나벨 얀켈,록키 모튼': 1,
     '이승무': 1,
     '스티븐 번스타인': 1,
     '찰스 스톤 3세': 1,
     '제임스 밴더빌트': 1,
     '니콜라스 메이어': 1,
     '켄 로네건': 1,
     '리사 촐로덴코': 1,
     '케빈 스페이시': 1,
     '제임스 쿨렌 브레삭': 1,
     '스테판 헤렉': 1,
     '마크 오스본,존 스티븐슨': 1,
     '스튜어트 옴': 1,
     '카를로스 아빌라': 1,
     '에릭 칼슨': 1,
     '마이클 그레이시': 1,
     '랜다 헤인즈': 1,
     '페테르 내스': 1,
     '로버트 론고': 1,
     '톰 고미칸': 1,
     '내쉬 에드게톤': 1,
     '조 핏카': 1,
     '캐리 커크패트릭,팀 존슨': 1,
     '에반 릭스,고든 헌트': 1,
     '다니엘 골드버그': 1,
     '조나스 엘머': 1,
     '울리히 에델': 1,
     '제임스 아이보리': 1,
     '줄리언 재롤드': 1,
     '앤드류 더글라스': 1,
     '로베르토 로드리게즈': 1,
     '루이스 모르뉴': 1,
     '조셉 고든 레빗': 1,
     '니콜라스 스톨러,더그 스윗랜드': 1,
     '라민 바흐러니': 1,
     '앤드류 댄': 1,
     '줄리언 템플': 1,
     '트레이 넬슨': 1,
     '스티브 마티노,마이크 트메이어': 1,
     '알버트 브룩스': 1,
     '존 D. 핸콕': 1,
     '아테 드 종': 1,
     'C.M. 토킹톤': 1,
     '로버트 이스코브': 1,
     '수잔나 화이트': 1,
     '커크 존스': 1,
     '페넬로페 스피리스': 1,
     '피터 스펜서': 1,
     '티보 타카스': 1,
     '진 와일더': 1,
     '마브룩 엘 메크리': 1,
     '에드 해리스': 1,
     '스탠리 도넌': 1,
     '말콤 벤빌': 1,
     '카렌 몬크리프': 1,
     '토베 후퍼': 1,
     '찰리 카우프만': 1,
     '폴커 슐렌도르프': 1,
     '존 체리': 1,
     '다니엘 스턴': 1,
     '그레그 스트라우스,콜린 스트라우스': 1,
     '마크 쉬홀르만': 1,
     '마크 러팔로': 1,
     '마이크 가브리엘,에릭 골드버그': 1,
     '빅터 살바': 1,
     '아비 콘': 1,
     '스티브 마티노': 1,
     '지미 헤이워드,스티브 마티노': 1,
     '마이클 노울즈': 1,
     '존 개틴스': 1,
     '카를로 구스타프': 1,
     '크리스찬 구드가스트': 1,
     '위트 스틸먼': 1,
     '데이비드 구겐하임': 1,
     '세스 고든': 1,
     '맥스 마이어': 1,
     '매튜 니나버': 1,
     '드로 소레프': 1,
     '로날드 님': 1,
     '폴 사벨라': 1,
     '조 차바닉': 1,
     '로아 우타우': 1,
     '크리스토퍼 스펜서': 1,
     '스티븐 엘리엇': 1,
     '조 루소,안소니 루소': 1,
     '피터 애튼시오': 1,
     '토마스 마이클 도넬리': 1,
     '피터 랜즈먼': 1,
     '존 루카스,스콧 무어': 1,
     '존 슬래터리': 1,
     '사이먼 웰스': 1,
     '니콜라스 맥카시': 1,
     '서극': 1,
     '루크 제이든': 1,
     '스테판 케이': 1,
     '제니퍼 챔버스 린치': 1,
     '빌 팩스톤': 1,
     '데이비드 젤너': 1,
     '밥 돌만': 1,
     '니콜라스 윈딩 레픈': 1,
     '크레이 보리스': 1,
     '에릭 카슨': 1,
     '크리스 로버츠': 1,
     '존 로버츠': 1,
     '알란 미터': 1,
     '다니엘 B. 이스크': 1,
     'R.J. 커들러': 1,
     '유렉 보가예비츠': 1,
     '브루스 로빈슨': 1,
     '마시밀리아노 세르치': 1,
     '데이비드 듀코브니': 1,
     '폴 질러': 1,
     '셀돈 레티츠': 1,
     '피터 닥터,데이빗 실버맨,리 언크리치': 1,
     '존 도일': 1,
     '야누즈 카민스키': 1,
     '안톤 코르빈': 1,
     '존 오트만': 1,
     '테론 R. 팔슨즈': 1,
     '라비 다르': 1,
     '노베르토 바르바': 1,
     '길 키넌': 1,
     '피터 스피어이그,마이클 스피어리그': 1,
     '톰 홀랜드': 1,
     '대런 린 보우즈만': 1,
     '닉 햄': 1,
     '브래들리 파커': 1,
     '비번 키드론': 1,
     '조시 쿨리': 1,
     '조슈아 마이클 스턴': 1,
     '마이클 슘웨이': 1,
     '그레고리 위덴': 1,
     '닐스 아르덴 오플레브': 1,
     '딘 셈러': 1,
     '데이비드 와일스': 1,
     '라스 클리브버그': 1,
     '로버트 차토프': 1,
     '니콜라스 자렉키': 1,
     '닐스 팀': 1,
     '로버트 스티븐슨': 1,
     '리차드 C. 사라피안': 1,
     '데이빗 M. 로젠탈': 1,
     '기예르모 델 토로': 1,
     '제이 챈드라세카': 1,
     '아담 브룩스': 1,
     '레리 하린': 1,
     '다니엘 페트리 주니어': 1,
     '로저 알러스': 1,
     '프랭크린 J. 샤프너': 1,
     '코린 하디': 1,
     '플로렝 에밀리오 시리': 1,
     '아니쉬 차간티': 1,
     '존 프랑켄하이머,리차드 스탠리': 1,
     '게리 그레버': 1,
     '잭 클레이튼': 1,
     '켄 마리노': 1,
     '데이빗 시겔,스콧 맥게히': 1,
     '발레리 페리스,조나단 데이턴': 1,
     '스티브 부세미': 1,
     '안소니 드라잔': 1,
     '랜스 훌': 1,
     '알렉스 매소네트': 1,
     '레이첼 탤러레이': 1,
     '헨리 윙클러': 1,
     '마이크 미첼,월트 도른': 1,
     '케이트 쉐아': 1,
     '윌 핀,댄 세인트 피에르': 1,
     '짐 펄': 1,
     '에이드리언 노블': 1,
     '캐시 얀': 1,
     '윌리엄 리에드': 1,
     '톰 바우건': 1,
     '그래그 참피온': 1,
     '조지 슬루이저': 1,
     '릭 비버': 1,
     '존 커랜': 1,
     '제다 토룬': 1,
     '빌 홀더먼': 1,
     '아리엘 브로멘': 1,
     '아그네츠카 보토위츠 보슬루': 1,
     '토니 밴크로프트,베리 쿡': 1,
     '오렌 무버만': 1,
     '다이안 커리스': 1,
     '피터 오팰론': 1,
     '브래들리 쿠퍼': 1,
     '케빈 리마,크리스 벅': 1,
     '조지 C. 울프': 1,
     '맥스 니콜스': 1,
     '존 크래신스키': 1,
     '프레데릭 와이즈만': 1,
     '파코 카베자스': 1,
     '하비에르 구티에레즈': 1,
     '파멜라 로만노프스키': 1,
     '마이클 골든버그': 1,
     '리 톨랜드 크리거': 1,
     '브라이언 피': 1,
     '윌 핀,제프리 카젠버그,돈 폴,비보 베르즈롱': 1,
     '케빈 소르보': 1,
     '테드 고체프': 1,
     '로저 알러스,롭 민코프': 1,
     '다렐 제임스 루트': 1,
     '킴벌리 피어스': 1,
     '할 니담': 1,
     '박찬욱': 1,
     '빅 암스트롱': 1,
     '메리 해론': 1,
     '프랭크 칼폰': 1,
     '자움 세라': 1,
     '에릭 이스튼버그': 1,
     '럭키 맥키': 1,
     '에드 게스-도넬리': 1,
     '빌 크로이어': 1,
     '마이클 도허티': 1,
     '댄 길로이': 1,
     '조나단 글레이저': 1,
     '딜런 키드': 1,
     '샘 펠,로버트 스티븐헤이겐': 1,
     '로버트 세이에': 1,
     '마이클 윈터바텀,존 커랜': 1,
     '해리 록': 1,
     '필리프 팔라도': 1,
     '게일 맨쿠소': 1,
     '알렌 휴즈,알버트 휴즈': 1,
     '브루노 바레토': 1,
     '돈 시겔': 1,
     '알렉산더 윗': 1,
     '나딤 사우마': 1,
     '스티븐 E. 드 수자': 1,
     '모튼 틸덤': 1,
     '라요스 콜타이': 1,
     '알폰소 아라우': 1,
     '뎁 헤이갠': 1,
     '다니엘 미나한': 1,
     '레오 가브리아제': 1,
     '크레이그 라이프': 1,
     '알레한드로 아그레스티': 1,
     '로엘 르네': 1,
     '레이첼 텔러레이': 1,
     '스티브 클로브스': 1,
     '콜린 스트로즈,그렉 스트로즈': 1,
     '스콧 모시어': 1,
     '존 듀이건': 1,
     '조셉 와트너체니': 1,
     '나이젤 콜': 1,
     '프라챠 핀카엡': 1,
     '제리 잭스': 1,
     '팀 블레이크 넬슨': 1,
     '찰리 카우프만,듀크 존슨': 1,
     '리즈 프리들랜더': 1,
     '데이빗 레이치,채드 스타헬스키': 1,
     '에릭 브리빅': 1,
     '스튜어트 헨들러': 1,
     '라마 모슬리': 1,
     '존 말루프,찰리 시스켈': 1,
     '브라이언 테일러,마크 네빌딘': 1,
     '애론 블레이즈,로버트 워커': 1,
     '올라턴드 오선샌미': 1,
     '크리스찬 세스마': 1,
     '타라 우드': 1,
     '데이브 메이어스': 1,
     '가이 그린': 1,
     '스티븐 T. 케이': 1,
     '제임스 마쉬': 1,
     '데이비스 구겐하임': 1,
     '레슬리 헤드랜드': 1,
     '세딕 니콜라스 트로얀': 1,
     '배리 젠킨스': 1,
     '데이브 파커': 1,
     '피터 솔레트': 1,
     '니콜라스 스톨러': 1,
     '브렌다 채프먼,스티브 히크너,사이먼 웰스': 1,
     '스튜어트 블럼버그': 1,
     '조나단 베이커,조쉬 베이커': 1,
     '데이빗 미어킨': 1,
     '로스 캐츠': 1,
     '피트 휴이트': 1,
     '카베 자헤디': 1,
     '코너 알린': 1,
     '미라 네이어': 1,
     '가이 퍼랜드': 1,
     '딜런 브라운': 1,
     '찰스 재롯': 1,
     '마이클 갤밴,세바스찬 바질': 1,
     '케빈 도노반': 1,
     '데이빗 프라이스': 1,
     '마이클 메이어': 1,
     '랜스 데일리': 1,
     '폴 다노': 1,
     '로버트 와이즈': 1,
     '스티븐 C. 밀러': 1,
     '로버츠 개너웨이': 1,
     '마이클 노어': 1,
     '앤드류 도미닉': 1,
     '레지나드 허드린,워링톤 허드린': 1,
     '짐 맥브라이드': 1,
     '존 해리슨': 1,
     '애나 릴리 아미푸르': 1,
     '마크 롭슨': 1,
     '빌 플림톤': 1,
     '루이스 모노': 1,
     '크리스 아이작슨': 1,
     '잭 헤일리 주니어': 1,
     '하모니 코린': 1,
     '데일 로너': 1,
     '루이 시호요스': 1,
     '말콤 D. 리': 1,
     '알렉산드라 딘': 1,
     '마크 앤드류스,브렌다 채프먼': 1,
     '크레이그 맥라켄': 1,
     '래리 찰스': 1,
     '하트 보크너': 1,
     '데이비드 셀처': 1,
     '안드레 외브레달': 1,
     '데릭 보트': 1,
     '오드리 웰스': 1,
     '존 화이트셀': 1,
     ...}




```python
for i in range(len(data)):
    for j in d_app:
        if j == data4.directors[i]:
            data4.director_appearance[i] = d_app[j]
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    


```python
data4
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>성인물(에로)</th>
      <th>스릴러</th>
      <th>애니메이션</th>
      <th>액션</th>
      <th>어드벤처</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117.0</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205.0</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813.0</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222.0</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824.0</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>NaN</td>
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
      <th>3666</th>
      <td>5819.0</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224.0</td>
      <td>개봉</td>
      <td>SF</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820.0</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821.0</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823.0</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126.0</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824.0</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 44 columns</p>
</div>



# 2-2. 감독 지난 3년간 매출액 평균


```python
pr_Y = data4.prdtYear
```


```python
data
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>showTypes</th>
      <th>audits</th>
      <th>Domestic</th>
      <th>Budget</th>
      <th>Distributor</th>
      <th>review</th>
      <th>MPAA</th>
      <th>stats</th>
      <th>raters</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>필름,2D</td>
      <td>청소년관람불가,청소년관람불가</td>
      <td>$35,061,555</td>
      <td>$15,000,000</td>
      <td>FilmDistrict</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.449755e+08</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>2D</td>
      <td>12세이상관람가</td>
      <td>$33,370,906</td>
      <td>$14,000,000</td>
      <td>Fox Searchlight Pictures</td>
      <td>This film was exceptional and one of the best ...</td>
      <td>12</td>
      <td>9.650832e+08</td>
      <td>250,206</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>2D</td>
      <td>청소년관람불가</td>
      <td>$291,477</td>
      <td>NaN</td>
      <td>Blue Fox Entertainment</td>
      <td>Strong performances, intimidating villains, cl...</td>
      <td>18</td>
      <td>9.096300e+06</td>
      <td>3,165</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>2D,4D</td>
      <td>청소년관람불가</td>
      <td>$92,029,184</td>
      <td>NaN</td>
      <td>Lionsgate</td>
      <td>In this 2nd installment of John Wick, the styl...</td>
      <td>18</td>
      <td>2.205583e+09</td>
      <td>351,231</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>필름,2D</td>
      <td>15세관람가,15세이상관람가</td>
      <td>$25,544,867</td>
      <td>$9,000,000</td>
      <td>Newmarket Films</td>
      <td>Thank Goodness I didn't read the reviews poste...</td>
      <td>15</td>
      <td>0.000000e+00</td>
      <td>1,096,788</td>
      <td>8.4</td>
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
      <th>3666</th>
      <td>5819</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224</td>
      <td>개봉</td>
      <td>SF</td>
      <td>NaN</td>
      <td>...</td>
      <td>필름</td>
      <td>중학생이상관람가</td>
      <td>$16,295,774</td>
      <td>$37,000,000</td>
      <td>Universal Pictures</td>
      <td>This is an original Marvel film that is actual...</td>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>42,255</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>필름</td>
      <td>중학생이상관람가</td>
      <td>$139,796</td>
      <td>NaN</td>
      <td>Cannon Film Distributors</td>
      <td>Hanna's War is the true story of Hanna Senesh,...</td>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>365</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>NaN</td>
      <td>...</td>
      <td>필름</td>
      <td>고등학생이상관람가</td>
      <td>$4,636,169</td>
      <td>NaN</td>
      <td>Paramount Pictures</td>
      <td>This courtroom thriller was one of the films t...</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>12,879</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td>NaN</td>
      <td>...</td>
      <td>필름</td>
      <td>중학생이상관람가</td>
      <td>$13,839,404</td>
      <td>NaN</td>
      <td>Paramount Pictures</td>
      <td>This is an excellent depiction of the insanity...</td>
      <td>18</td>
      <td>0.000000e+00</td>
      <td>23,328</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>필름</td>
      <td>연소자관람가</td>
      <td>$16,093,651</td>
      <td>NaN</td>
      <td>New Line Cinema</td>
      <td>Mind you, it's not supposed to be, but it is. ...</td>
      <td>12</td>
      <td>0.000000e+00</td>
      <td>5,794</td>
      <td>4.4</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 21 columns</p>
</div>




```python
data4['prdtYear'] = pr_Y
data4['director_year'] = pd.Series()

for i in range(len(data)):
    data4['director_year'][i] = str(data.directors[i])+ str(data.prdtYear[i])
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:670: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_with_indexer(indexer, value)
    


```python
data4
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>스릴러</th>
      <th>애니메이션</th>
      <th>액션</th>
      <th>어드벤처</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>director_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117.0</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>니콜라스 윈딩 레픈2011.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205.0</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>타이카 와이티티2019.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813.0</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>말릭 베이더2019.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222.0</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>채드 스타헬스키2017.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824.0</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>크리스토퍼 놀란2000.0</td>
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
      <th>3666</th>
      <td>5819.0</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224.0</td>
      <td>개봉</td>
      <td>SF</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>nan1986.0</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820.0</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>메나헴 골란1988.0</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821.0</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>nan1987.0</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823.0</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126.0</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>nan1987.0</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824.0</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>토마스 J 웨이드1989.0</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 45 columns</p>
</div>




```python
data4.Domestic = data4.Domestic.apply(lambda x: x.replace("$", ""))
data4.Domestic = data4.Domestic.apply(lambda x: x.replace(",", ""))
data4.Domestic = data4.Domestic.astype('int')
```


```python
d2 = {}
for i in range(len(data)):
    d2[data4.director_year[i]] = data4.Domestic[i]
```


```python
data4.directors.fillna("", inplace = True)
```


```python
type(d2[str(data4.directors[1]) + str(data4.prdtYear[1]-2)])
```




    numpy.int64




```python
for i in range(len(data4)):
    if data4.directors[i] != "":
        a = 0
        b = 0
        if str(data4.directors[i])+ str(data4.prdtYear[i]) in d2:
            if str(d2[str(data4.directors[i]) + str(data4.prdtYear[i])])!='nan':
                a+=d2[str(data4.directors[i]) + str(data4.prdtYear[i])]
        if str(data4.directors[i])+ str(data4.prdtYear[i]-1) in d2:
            if str(d2[str(data4.directors[i]) + str(data4.prdtYear[i]-1)])!='nan':
                a+=d2[str(data4.directors[i]) + str(data4.prdtYear[i]-1)]
        if str(data4.directors[i])+ str(data4.prdtYear[i]-2) in d2:
            if str(d2[str(data4.directors[i]) + str(data4.prdtYear[i]-2)])!='nan':
                a+=d2[str(data4.directors[i]) + str(data4.prdtYear[i]-2)]
        data4['director_revenue'][i] = a / 3
    else: 
        data4['director_revenue'][i] = 0
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      app.launch_new_instance()
    


```python
data4.drop('director_year',axis = 1, inplace = True)
```

# 3. 제작사 시장점유율로 수치화


```python
rank = pd.read_excel('data/영화사 순위.xlsx')
```


```python
d = {}
Dis = rank.Distributor
rank = rank.Share

for i in range(len(Dis)):
    d[Dis[i]] = rank[i]
```


```python
data5 = data4.copy()
```


```python
data5['distributor_share'] = pd.Series()
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      """Entry point for launching an IPython kernel.
    


```python
for i in range(len(data5)):
    tmp = data5.Distributor[i]
    for j in d:
        if type(tmp) != float :
            if j in tmp:           
                data5['distributor_share'][i] = d[j]
        else:
            data5['distributor_share'][i] = 0

```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


```python
data5
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>스릴러</th>
      <th>애니메이션</th>
      <th>액션</th>
      <th>어드벤처</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117.0</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.16872e+07</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205.0</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.16143e+08</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813.0</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>97159</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222.0</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>3.06764e+07</td>
      <td>0.0407</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824.0</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>8.51496e+06</td>
      <td>0.0020</td>
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
      <th>3666</th>
      <td>5819.0</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224.0</td>
      <td>개봉</td>
      <td>SF</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1173</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820.0</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.13221e+07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821.0</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823.0</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126.0</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824.0</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>5.36455e+06</td>
      <td>0.0264</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 45 columns</p>
</div>




```python
for i in range(len(data5)):
    if type(data5.Distributor[i]) != float:
        if data5.Distributor[i] == 'Twentieth Century Fox':
            data5.distributor_share[i] = 0.1104
        if data5.Distributor[i].__contains__('DreamWorks'):
            data5.distributor_share[i] = 0.0183
        if data5.Distributor[i] == 'The Weinstein Company':
            data5.distributor_share[i] = 0.0094
        if data5.Distributor[i].__contains__('Paramount'):
            data5.distributor_share[i] = 0.1035
        if data5.Distributor[i].__contains__('Maya'):
            data5.distributor_share[i] = 0
        if data5.Distributor[i].__contains__('TriStar'):
            data5.distributor_share[i] = 0.0005  
        if data5.Distributor[i].__contains__('Abramorama'):
            data5.distributor_share[i] = 0.0001
        if data5.Distributor[i].__contains__('Wellspring'):
            data5.distributor_share[i] = 0.0001
        if data5.Distributor[i] == 'IMAX':
            data5.distributor_share[i] = 0.0014
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      app.launch_new_instance()
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':
    


```python
data5.distributor_share.fillna(0,inplace = True)
```


```python
data5
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>스릴러</th>
      <th>애니메이션</th>
      <th>액션</th>
      <th>어드벤처</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117.0</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.16872e+07</td>
      <td>0.0016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205.0</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.16143e+08</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813.0</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>97159</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222.0</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>3.06764e+07</td>
      <td>0.0407</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824.0</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>8.51496e+06</td>
      <td>0.0020</td>
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
      <th>3666</th>
      <td>5819.0</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224.0</td>
      <td>개봉</td>
      <td>SF</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1173</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820.0</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.13221e+07</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821.0</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823.0</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126.0</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824.0</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>5.36455e+06</td>
      <td>0.0264</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 45 columns</p>
</div>



# 4. openDt 계절별, 분기별

# 4-1. 분기별


```python
data5['month'] = data5.openDt.apply(lambda x: x % 10000)
```


```python
data5.month = data5.month.apply(lambda x : x //100)
```


```python
data5['openDt_quarter'] = data5.month.apply(lambda x: 1 if x <=3 else 2 if x <=6 else 3 if x <=9 else 4)
```

# 4-2. 계절별


```python
data5['openDt_weather'] = data5.month.apply(lambda x: "winter" if x<=2 or x == 12 else "spring" if x <=5 else "summer" if x <=8 else "fall")
```

# 5. prdtYear와 openDt의 차이 (년도)


```python
openY = data5.openDt.apply(lambda x: int(str(x)[0:4]))
```


```python
data5['year_gap'] = openY - data5.prdtYear
```


```python
data5
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
      <th>index</th>
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>...</th>
      <th>전쟁</th>
      <th>코미디</th>
      <th>판타지</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
      <th>month</th>
      <th>openDt_quarter</th>
      <th>openDt_weather</th>
      <th>year_gap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117.0</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.16872e+07</td>
      <td>0.0016</td>
      <td>11.0</td>
      <td>4</td>
      <td>fall</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205.0</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.16143e+08</td>
      <td>0.0000</td>
      <td>2.0</td>
      <td>1</td>
      <td>winter</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813.0</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>97159</td>
      <td>0.0000</td>
      <td>8.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222.0</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>3.06764e+07</td>
      <td>0.0407</td>
      <td>2.0</td>
      <td>1</td>
      <td>winter</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824.0</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>8.51496e+06</td>
      <td>0.0020</td>
      <td>8.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1.0</td>
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
      <th>3666</th>
      <td>5819.0</td>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224.0</td>
      <td>개봉</td>
      <td>SF</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1173</td>
      <td>12.0</td>
      <td>4</td>
      <td>winter</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>5820.0</td>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.13221e+07</td>
      <td>0.0000</td>
      <td>10.0</td>
      <td>4</td>
      <td>fall</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>5821.0</td>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td></td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
      <td>11.0</td>
      <td>4</td>
      <td>fall</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>5823.0</td>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126.0</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td></td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
      <td>1.0</td>
      <td>1</td>
      <td>winter</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>5824.0</td>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>5.36455e+06</td>
      <td>0.0264</td>
      <td>8.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 49 columns</p>
</div>



## 5-1. year_gap 상한선(98%) 하한선(0)


```python
data5.year_gap.value_counts()
```




    0.0     1656
    1.0     1388
    2.0      337
    3.0      121
    4.0       54
    5.0       23
    6.0       16
    7.0       14
    8.0       11
    9.0        9
    13.0       8
    14.0       4
    18.0       3
    11.0       3
    17.0       3
    33.0       2
    22.0       2
    30.0       2
    52.0       1
    73.0       1
    10.0       1
    12.0       1
    15.0       1
    26.0       1
    16.0       1
    20.0       1
    58.0       1
    28.0       1
    32.0       1
    57.0       1
    38.0       1
    31.0       1
    43.0       1
    Name: year_gap, dtype: int64




```python
np.percentile(data5.year_gap, 98)
```




    7.0




```python
data5.year_gap = data5.year_gap.apply(lambda x: 7 if x > 7 else x)
```

# 6. showTypes 개수로 정제


```python
data5['showTypes_num'] = data5['showTypes'].apply(lambda x : len(list(set(str(x).split(',')))))
```


```python
data5[['showTypes','showTypes_num']].head(60)
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
      <th>showTypes</th>
      <th>showTypes_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2D,4D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>필름,2D,4D,IMAX</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2D,IMAX</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2D,4D,IMAX</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2D,2D,4D,4D,IMAX</td>
      <td>3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>필름,2D,2D,4D,IMAX</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>필름,2D,4D,IMAX</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2D,2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2D,IMAX,ScreenX</td>
      <td>3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2D,3D,4D,IMAX,ScreenX</td>
      <td>5</td>
    </tr>
    <tr>
      <th>46</th>
      <td>필름,2D,3D,4D,IMAX</td>
      <td>5</td>
    </tr>
    <tr>
      <th>47</th>
      <td>필름,2D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>필름,2D,3D,4D,IMAX</td>
      <td>5</td>
    </tr>
    <tr>
      <th>50</th>
      <td>필름,2D,3D,4D,IMAX</td>
      <td>5</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2D,4D</td>
      <td>2</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2D,4D,IMAX</td>
      <td>3</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2D,4D,IMAX</td>
      <td>3</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2D</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2D</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data5.drop('showTypes',axis = 1,inplace = True)
```

# 7. 배우 정제


```python
ac_r = pd.read_excel('data/영화배우 순위.xlsx')
```


```python
ac_r.Name = ac_r.Name.apply(lambda x: x.lower())
```


```python
ac_r['Name_year'] = pd.Series()
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      """Entry point for launching an IPython kernel.
    


```python
for i in range(len(ac_r)):
    ac_r['Name_year'][i] = ac_r.Name[i] + str(ac_r.year[i])
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:670: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_with_indexer(indexer, value)
    


```python
ac_dict = ac_r.set_index('Name_year')['Star'].to_dict()
```


```python
en = pd.read_excel('data/1970_2019_배우.xlsx')
```


```python
en.dropna(subset = ['director_en'], inplace = True)
```


```python
en.director_en = en.director_en.apply(lambda x: x.lower())
```


```python
en_dict = en.set_index('director')['director_en'].to_dict()
```


```python
data5.actors = data5.actors.str.split(',')
```


```python
def to_en(x):
    if type(x) == list :
        for i in range(len(x)):
            try:
                x[i] = en_dict[x[i]]
            except:
                x[i] = ""
    return x
```


```python
data5['actors'] = data5['actors'].apply(lambda x: to_en(x))
```


```python
data5['Name_year'] = pd.Series()
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      """Entry point for launching an IPython kernel.
    


```python
for x in range(len(data5.movieCd)):
    ny = []
    try:
        for i in range(len(data5.actors[x])):
            ny.append(data5.actors[x][i] + str(int(data5.prdtYear[x])))
            data5['Name_year'][x] = ny
    except: pass
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


```python
def to_score(x):
    if type(x) == list :
        for i in range(len(x)):
            try:
                x[i] = ac_dict[x[i]]
            except:
                x[i] = 0
    return x
```


```python
data5['Name_year'].apply(lambda x: to_score(x))
```




    0                                  [104, 0]
    1       [262, 25, 25, 25, 121, 184, 25, 71]
    2                                      [44]
    3        [66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    4                                [0, 0, 96]
                           ...                 
    3666                                 [0, 0]
    3667                                    NaN
    3668                                 [0, 0]
    3669                                    [0]
    3670                                    [0]
    Name: Name_year, Length: 3671, dtype: object




```python
def mean(x):
    try:
        x = round(sum(x,0)/len(x),2)
    except:
        x = 0
    return x
```


```python
data5['actor_score'] = data5.Name_year.apply(lambda x: mean(x))
```


```python
data5.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3671 entries, 0 to 3670
    Data columns (total 51 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   index                3671 non-null   float64
     1   movieCd              3671 non-null   object 
     2   movieNm              3671 non-null   object 
     3   movieNmEn            3671 non-null   object 
     4   showTm               3671 non-null   float64
     5   prdtYear             3671 non-null   float64
     6   openDt               3671 non-null   float64
     7   prdtStatNm           3671 non-null   object 
     8   genres               3671 non-null   object 
     9   directors            3671 non-null   object 
     10  actors               3536 non-null   object 
     11  audits               3644 non-null   object 
     12  Domestic             3671 non-null   int64  
     13  Budget               1908 non-null   object 
     14  Distributor          3659 non-null   object 
     15  review               3312 non-null   object 
     16  MPAA                 2774 non-null   object 
     17  stats                3474 non-null   float64
     18  raters               3360 non-null   object 
     19  ratings              3360 non-null   float64
     20  SF                   3671 non-null   float64
     21  가족                   3671 non-null   float64
     22  공연                   3671 non-null   float64
     23  공포(호러)               3671 non-null   float64
     24  기타                   3671 non-null   float64
     25  다큐멘터리                3671 non-null   float64
     26  드라마                  3671 non-null   float64
     27  멜로/로맨스               3671 non-null   float64
     28  뮤지컬                  3671 non-null   float64
     29  미스터리                 3671 non-null   float64
     30  범죄                   3671 non-null   float64
     31  사극                   3671 non-null   float64
     32  서부극(웨스턴)             3671 non-null   float64
     33  성인물(에로)              3671 non-null   float64
     34  스릴러                  3671 non-null   float64
     35  애니메이션                3671 non-null   float64
     36  액션                   3671 non-null   float64
     37  어드벤처                 3671 non-null   float64
     38  전쟁                   3671 non-null   float64
     39  코미디                  3671 non-null   float64
     40  판타지                  3671 non-null   float64
     41  director_appearance  3548 non-null   object 
     42  director_revenue     3671 non-null   object 
     43  distributor_share    3671 non-null   float64
     44  month                3671 non-null   float64
     45  openDt_quarter       3671 non-null   int64  
     46  openDt_weather       3671 non-null   object 
     47  year_gap             3671 non-null   float64
     48  showTypes_num        3671 non-null   int64  
     49  Name_year            3536 non-null   object 
     50  actor_score          3671 non-null   float64
    dtypes: float64(31), int64(3), object(17)
    memory usage: 1.6+ MB
    


```python
data5.drop(['index'],axis=1, inplace = True)
```


```python
data5.drop(['actors'],axis=1, inplace = True)
```


```python
data5.drop(['Name_year'],axis=1, inplace = True)
```


```python
data5.drop(['Budget'],axis=1, inplace = True)
```


```python
data5.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3671 entries, 0 to 3670
    Data columns (total 47 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   movieCd              3671 non-null   object 
     1   movieNm              3671 non-null   object 
     2   movieNmEn            3671 non-null   object 
     3   showTm               3671 non-null   float64
     4   prdtYear             3671 non-null   float64
     5   openDt               3671 non-null   float64
     6   prdtStatNm           3671 non-null   object 
     7   genres               3671 non-null   object 
     8   directors            3671 non-null   object 
     9   audits               3644 non-null   object 
     10  Domestic             3671 non-null   int64  
     11  Distributor          3659 non-null   object 
     12  review               3312 non-null   object 
     13  MPAA                 2774 non-null   object 
     14  stats                3474 non-null   float64
     15  raters               3360 non-null   object 
     16  ratings              3360 non-null   float64
     17  SF                   3671 non-null   float64
     18  가족                   3671 non-null   float64
     19  공연                   3671 non-null   float64
     20  공포(호러)               3671 non-null   float64
     21  기타                   3671 non-null   float64
     22  다큐멘터리                3671 non-null   float64
     23  드라마                  3671 non-null   float64
     24  멜로/로맨스               3671 non-null   float64
     25  뮤지컬                  3671 non-null   float64
     26  미스터리                 3671 non-null   float64
     27  범죄                   3671 non-null   float64
     28  사극                   3671 non-null   float64
     29  서부극(웨스턴)             3671 non-null   float64
     30  성인물(에로)              3671 non-null   float64
     31  스릴러                  3671 non-null   float64
     32  애니메이션                3671 non-null   float64
     33  액션                   3671 non-null   float64
     34  어드벤처                 3671 non-null   float64
     35  전쟁                   3671 non-null   float64
     36  코미디                  3671 non-null   float64
     37  판타지                  3671 non-null   float64
     38  director_appearance  3548 non-null   object 
     39  director_revenue     3671 non-null   object 
     40  distributor_share    3671 non-null   float64
     41  month                3671 non-null   float64
     42  openDt_quarter       3671 non-null   int64  
     43  openDt_weather       3671 non-null   object 
     44  year_gap             3671 non-null   float64
     45  showTypes_num        3671 non-null   int64  
     46  actor_score          3671 non-null   float64
    dtypes: float64(30), int64(3), object(14)
    memory usage: 1.5+ MB
    


```python
data5
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
      <th>movieCd</th>
      <th>movieNm</th>
      <th>movieNmEn</th>
      <th>showTm</th>
      <th>prdtYear</th>
      <th>openDt</th>
      <th>prdtStatNm</th>
      <th>genres</th>
      <th>directors</th>
      <th>audits</th>
      <th>...</th>
      <th>판타지</th>
      <th>director_appearance</th>
      <th>director_revenue</th>
      <th>distributor_share</th>
      <th>month</th>
      <th>openDt_quarter</th>
      <th>openDt_weather</th>
      <th>year_gap</th>
      <th>showTypes_num</th>
      <th>actor_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117.0</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>청소년관람불가,청소년관람불가</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.16872e+07</td>
      <td>0.0016</td>
      <td>11.0</td>
      <td>4</td>
      <td>fall</td>
      <td>0.0</td>
      <td>2</td>
      <td>52.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20197390</td>
      <td>조조 래빗</td>
      <td>JOJO RABBIT</td>
      <td>108.0</td>
      <td>2019.0</td>
      <td>20200205.0</td>
      <td>개봉</td>
      <td>코미디,드라마,전쟁</td>
      <td>타이카 와이티티</td>
      <td>12세이상관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.16143e+08</td>
      <td>0.0000</td>
      <td>2.0</td>
      <td>1</td>
      <td>winter</td>
      <td>1.0</td>
      <td>1</td>
      <td>92.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20203227</td>
      <td>킬러맨</td>
      <td>Killerman</td>
      <td>112.0</td>
      <td>2019.0</td>
      <td>20200813.0</td>
      <td>개봉</td>
      <td>액션,범죄,드라마</td>
      <td>말릭 베이더</td>
      <td>청소년관람불가</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>97159</td>
      <td>0.0000</td>
      <td>8.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1.0</td>
      <td>1</td>
      <td>44.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20166101</td>
      <td>존 윅 - 리로드</td>
      <td>John Wick Chapter Two</td>
      <td>122.0</td>
      <td>2017.0</td>
      <td>20170222.0</td>
      <td>개봉</td>
      <td>액션,범죄,스릴러</td>
      <td>채드 스타헬스키</td>
      <td>청소년관람불가</td>
      <td>...</td>
      <td>0.0</td>
      <td>2</td>
      <td>3.06764e+07</td>
      <td>0.0407</td>
      <td>2.0</td>
      <td>1</td>
      <td>winter</td>
      <td>0.0</td>
      <td>2</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20010238</td>
      <td>메멘토</td>
      <td>Memento</td>
      <td>113.0</td>
      <td>2000.0</td>
      <td>20010824.0</td>
      <td>개봉</td>
      <td>미스터리,범죄,스릴러</td>
      <td>크리스토퍼 놀란</td>
      <td>15세관람가,15세이상관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>5</td>
      <td>8.51496e+06</td>
      <td>0.0020</td>
      <td>8.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1.0</td>
      <td>2</td>
      <td>32.00</td>
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
      <th>3666</th>
      <td>19870151</td>
      <td>하워드 덕</td>
      <td>Howard The Duck</td>
      <td>110.0</td>
      <td>1986.0</td>
      <td>19871224.0</td>
      <td>개봉</td>
      <td>SF</td>
      <td></td>
      <td>중학생이상관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1173</td>
      <td>12.0</td>
      <td>4</td>
      <td>winter</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3667</th>
      <td>19880198</td>
      <td>한나스 워</td>
      <td>Hanna'S War</td>
      <td>145.0</td>
      <td>1988.0</td>
      <td>19881015.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>메나헴 골란</td>
      <td>중학생이상관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.13221e+07</td>
      <td>0.0000</td>
      <td>10.0</td>
      <td>4</td>
      <td>fall</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3668</th>
      <td>19870145</td>
      <td>핫스 오브 화이어</td>
      <td>Hearts Of Fire</td>
      <td>100.0</td>
      <td>1987.0</td>
      <td>19871128.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td></td>
      <td>고등학생이상관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
      <td>11.0</td>
      <td>4</td>
      <td>fall</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>19900127</td>
      <td>햄버거 힐</td>
      <td>Hamberger Hill</td>
      <td>109.0</td>
      <td>1987.0</td>
      <td>19900126.0</td>
      <td>개봉</td>
      <td>드라마,액션,전쟁</td>
      <td></td>
      <td>중학생이상관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.1035</td>
      <td>1.0</td>
      <td>1</td>
      <td>winter</td>
      <td>3.0</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>19900258</td>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td>No Holds Barred</td>
      <td>92.0</td>
      <td>1989.0</td>
      <td>19900804.0</td>
      <td>개봉</td>
      <td>드라마</td>
      <td>토마스 J 웨이드</td>
      <td>연소자관람가</td>
      <td>...</td>
      <td>0.0</td>
      <td>1</td>
      <td>5.36455e+06</td>
      <td>0.0264</td>
      <td>8.0</td>
      <td>3</td>
      <td>summer</td>
      <td>1.0</td>
      <td>1</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>3671 rows × 47 columns</p>
</div>



# 8. MPAA null 채우기
### 장르별 최빈값


```python
data6 = data[['genres','MPAA']]
```


```python
data6['genre'] = data6.genres.fillna('기타').apply(lambda x: x.split(',')[0])
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
data6.MPAA = data6.MPAA.replace('all',0)
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py:5303: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self[name] = value
    


```python
# 장르별 최빈값 확인

#data7 = data6.dropna(subset = ['MPAA'])
#data7.MPAA = data7.MPAA.astype('int')
#data8 = data7.pivot_table(index = ['genre','MPAA'], aggfunc = 'count')
#data8.to_excel('genre.xlsx')
```


```python
# SF = 12
# 가족 = 12
# 공연 = 18
# 공포(호러) = 18
# 기타 = 15
# 다큐멘터리 = 15
# 드라마 = 15
# 멜로/로맨스 = 15
# 뮤지컬 = 15
# 미스터리 = 15
# 범죄 = 18
# 사극 = 15
# 서부극(웨스턴) = 15
# 성인물(에로) = 18
# 스릴러 = 15
# 애니메이션 = 12
# 액션 = 15
# 어드벤처 = 12
# 전쟁 = 15
# 코미디 = 15
# 판타지 = 12

#'SF', '가족', '공연', '공포(호러)', '기타', '다큐멘터리', '드라마', '멜로/로맨스', '뮤지컬',
#       '미스터리', '범죄', '사극', '서부극(웨스턴)', '성인물(에로)', '스릴러', '애니메이션', '액션', '어드벤처',
#       '전쟁', '코미디', '판타지'
```


```python
genre_MPAA = {'SF':12,
              '가족':12, 
              '공연':18, 
              '공포(호러)':18, 
              '기타':15, 
              '다큐멘터리':15, 
              '드라마':15,
              '멜로/로맨스':15, 
              '뮤지컬':15,
              '미스터리':15,
              '범죄':18,
              '사극':15,
              '서부극(웨스턴)':15,
              '성인물(에로)':18, 
              '스릴러':15, 
              '애니메이션':12, 
              '액션':15, 
              '어드벤처':12,
              '전쟁':15,
              '코미디':15,
              '판타지':12}
```


```python
for i in range(len(data6)):
    if type(data6.MPAA[i]) == float:
        data6.MPAA[i] = genre_MPAA[data6.genre[i]]
```

    /Users/injin/opt/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3331: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      exec(code_obj, self.user_global_ns, self.user_ns)
    


```python
data5.MPAA = data6.MPAA
```

# 9. 컬럼명 변경 (필요하면)


```python
data5.rename(columns = {'가족':'family', 
                        '공연':'performance', 
                        '공포(호러)':'horror',
                        '기타':'etc', 
                        '다큐멘터리':'documentary', 
                        '드라마':'drama', 
                        '멜로/로맨스':'romance', 
                        '뮤지컬':'musical', 
                        '미스터리':'mystery', 
                        '범죄':'crime', 
                        '사극':'history', 
                        '서부극(웨스턴)':'western',
                        '성인물(에로)':'adult', 
                        '스릴러':'thriller', 
                        '애니메이션':'animation', 
                        '액션':'actioin', 
                        '어드벤처':'adventure', 
                        '전쟁':"war", 
                        '코미디':"comedy", 
                        '판타지':"fantasy"}, inplace = True)
```


```python
# data5 = data5.rename({'stats':"kor_sales"},axis = 'columns')
```

# 10. 데이터 저장


```python
data5.to_excel('정제test.xlsx')
```
