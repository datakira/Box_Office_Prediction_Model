```
!pip install jpype1==0.7.0
!pip install konlpy

```

    Collecting jpype1==0.7.0
    [?25l  Downloading https://files.pythonhosted.org/packages/07/09/e19ce27d41d4f66d73ac5b6c6a188c51b506f56c7bfbe6c1491db2d15995/JPype1-0.7.0-cp36-cp36m-manylinux2010_x86_64.whl (2.7MB)
    [K     |████████████████████████████████| 2.7MB 5.2MB/s 
    [?25hInstalling collected packages: jpype1
    Successfully installed jpype1-0.7.0
    Collecting konlpy
    [?25l  Downloading https://files.pythonhosted.org/packages/85/0e/f385566fec837c0b83f216b2da65db9997b35dd675e107752005b7d392b1/konlpy-0.5.2-py2.py3-none-any.whl (19.4MB)
    [K     |████████████████████████████████| 19.4MB 1.4MB/s 
    [?25hCollecting beautifulsoup4==4.6.0
    [?25l  Downloading https://files.pythonhosted.org/packages/9e/d4/10f46e5cfac773e22707237bfcd51bbffeaf0a576b0a847ec7ab15bd7ace/beautifulsoup4-4.6.0-py3-none-any.whl (86kB)
    [K     |████████████████████████████████| 92kB 8.1MB/s 
    [?25hCollecting tweepy>=3.7.0
      Downloading https://files.pythonhosted.org/packages/bb/7c/99d51f80f3b77b107ebae2634108717362c059a41384a1810d13e2429a81/tweepy-3.9.0-py2.py3-none-any.whl
    Requirement already satisfied: numpy>=1.6 in /usr/local/lib/python3.6/dist-packages (from konlpy) (1.18.5)
    Requirement already satisfied: JPype1>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (0.7.0)
    Collecting colorama
      Downloading https://files.pythonhosted.org/packages/c9/dc/45cdef1b4d119eb96316b3117e6d5708a08029992b2fee2c143c7a0a5cc5/colorama-0.4.3-py2.py3-none-any.whl
    Requirement already satisfied: lxml>=4.1.0 in /usr/local/lib/python3.6/dist-packages (from konlpy) (4.2.6)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.3.0)
    Requirement already satisfied: requests[socks]>=2.11.1 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (2.23.0)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from tweepy>=3.7.0->konlpy) (1.15.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib>=0.7.0->tweepy>=3.7.0->konlpy) (3.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2020.6.20)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (2.10)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6; extra == "socks" in /usr/local/lib/python3.6/dist-packages (from requests[socks]>=2.11.1->tweepy>=3.7.0->konlpy) (1.7.1)
    Installing collected packages: beautifulsoup4, tweepy, colorama, konlpy
      Found existing installation: beautifulsoup4 4.6.3
        Uninstalling beautifulsoup4-4.6.3:
          Successfully uninstalled beautifulsoup4-4.6.3
      Found existing installation: tweepy 3.6.0
        Uninstalling tweepy-3.6.0:
          Successfully uninstalled tweepy-3.6.0
    Successfully installed beautifulsoup4-4.6.0 colorama-0.4.3 konlpy-0.5.2 tweepy-3.9.0
    


```
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
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


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

```

# Story To PCA


```
df = pd.read_excel("/content/drive/My Drive/Data/1970_2019_영화전체.xlsx")
df.스토리_ko.fillna("", inplace=True)
df.스토리_ko = df.스토리_ko.astype(str)
df.genres = df.genres.astype(str)

df.스토리_ko = df.스토리_ko.apply(lambda x:x.replace("\n"," "))
df.스토리_ko = df.스토리_ko.progress_apply(trim_ko)
```


    HBox(children=(FloatProgress(value=0.0, max=5825.0), HTML(value='')))


    
    


```
corpus = df.스토리_ko[(df.스토리_ko.notnull()) & (df.스토리_ko.apply(lambda x: True if len(x)>3 else False))]
corpus = corpus.reset_index()
corpus.head()
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
      <th>스토리_ko</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>줄거리 내 생각 들린다고 자신 개발 텔레파시 장치 통해 강아지 헨리 의 생각 읽을 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>줄거리 인류 위협 하는 공룡 맞서 싸우는 특수 대원 활약 그린 판타지 액션 영화</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>줄거리 월요일 아침 학교 늦은 아들 데려다 주고 출근 해야하는 레이첼 꽉 막힌 도로...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>줄거리 제 차 세계대전 말기 엄마 로지 스칼렛 요한슨 와 단둘 살 있는 살 소년 조...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>줄거리 당신 줄 건 한 단어 테넷 이해 하지 느껴라 시간 흐름 뒤집는 인 버전 통해...</td>
    </tr>
  </tbody>
</table>
</div>




```
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(corpus.스토리_ko.values)
X = X.toarray()
X.shape
```




    (2676, 2000)




```
pca = PCA(n_components=5)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

new_X = pca.fit_transform(X)
new_X.shape
```

    [0.00719843 0.00636207 0.00560346 0.00490829 0.00445288]
    [4.3205935  4.06184837 3.81199981 3.56771328 3.39816956]
    




    (2676, 5)




```
ret = most_similar(new_X, 0, 10, method="cosine")
for i in ret:
    index = corpus.loc[i[1],'index']
    print(f" {df.loc[index, 'genres']:10s} : {df.loc[index, '스토리_ko'][:100]:}")
    #print(f"{df.loc[i[1], '스토리_ko'][:100]:}")
```

     코미디,드라마    : 줄거리 내 생각 들린다고 자신 개발 텔레파시 장치 통해 강아지 헨리 의 생각 읽을 수 있게 된 천재 소년 올리버 성공 대한 기쁨 잠시 가족 떨어져 살 될 위기 처 올리버 와 헨리 
     애니메이션,판타지,뮤지컬,어드벤처,가족 : 줄거리 오즈 마법사 원작 탄생 주년 기념 대작 빛 사라진 에메랄드 시티 구원 할 도로시 돌아왔다 안녕 나 캔자스 사는 도로시 해 날 마법 같은 일어났어 무지개 나를 쫓아와 어디 론
     공포(호러),스릴러 : 줄거리 기억 너머 숨어있던 거대한 진실 당신 감각 지배 할 초 현실 로 스릴러 온다 모범생 베케트 파티 광 럭키 천재 엘리엇 반항 레이브 이 네 명의 남녀 베케트 부모님 방치 해 
     애니메이션,코미디,어드벤처 : 줄거리 심심한 세상 끝났다 곧 하늘 무너지고 내 뜬다 슬슬 세상 구 해볼까 갑자기 하늘 떨어진 무언가 의해 머리 강타 치킨 리틀 하늘 무너지고 있다고 확신 고향 마을 오우 키 오크
     애니메이션,어드벤처 : 줄거리 니코 떠나는 신나는 어드벤처 가자 신비한 마법 세계 할아버지 할머니 살 있는 살 니코 친구 괴롭힘 당하다가 실수 오래된 나무 불 지르게 되고 성난 나무 니코 할아버지 할머니
     애니메이션,어드벤처,가족,판타지 : 줄거리 과자 먹으면 동물 변신 반복 되는 일상 지루해하던 오웬 신비한 과자 상자 삼촌 유품 남겨진다 세상 먹는 순간 동물 변하는 마법 과자 평생 꿈꿔 온 서커스 시작 해보기로 결심
     애니메이션,가족,어드벤처,코미디 : 줄거리 이 세상 가장 특별한 가족 온다 치즈 마을 지하 마을 사람 상상 할 존재 살 있다 그 바로 네모 반듯한 박스 입고 다니는 귀여운 몬스터 박스 트롤 박스 쓴 인간 소년 비록 
     스릴러        : 줄거리 가장 어 두운 악 깨우는 가장 빛나는 능력 샤이닝 먹어라 그 막아라 어린시절 아버지 남긴 트라우마 벗어나지 못 대니 자신 가진 샤이닝 능력 죽음 앞둔 이 돕는 닥터 슬립 불
     액션         : 줄거리 모든 히어로 능력 하나로 모였다 솔로몬 지혜 헤라클레스 힘 아틀라스 체력 제우스 권능 아킬레스 용기 머큐리 스피드 슈퍼 파워 얻게 된 소년 자신 능력 깨닫고 악당 물리치는 
     액션,공포(호러)  : 줄거리 세상 종말 가져올 죽음 저주 깨어난다 잃어버린 도시 카르 를 찾아다니는 트레져 헌터 다니엘 와 노아 는 사막 한가운데 고대 미라 잠들어있는 신비로운 무덤 발견 한다 미라 주
    


```
kmeans = KMeans(n_clusters=10, random_state=0).fit(new_X)
kmeans.labels_
```




    array([2, 7, 9, ..., 5, 6, 1], dtype=int32)




```
data = pd.DataFrame([corpus['index'], new_X[:,0], new_X[:,1], new_X[:,2],new_X[:,3], new_X[:,4], kmeans.labels_]).T
data.columns=['df_index','x0','x1','x2','x3','x4','label']
data.df_index = data.df_index.astype(int)
data.label = data.label.astype(int)
```


```
f, ax = plt.subplots(2,2, figsize=(10,10))
for i in range(2):
    for j in range(2):
        if i==j:
            j2 = j+2
        else:
            j2 = j
        sns.scatterplot(x='x'+str(i),y='x'+str(j2),hue='label', data=data,
                legend='full', palette=sns.color_palette("gnuplot2_r", n_colors=data.label.nunique()),
                alpha=0.7, ax =ax[i][j])
        ax[i][j].set_title("x"+str(i)+" ~ x"+str(j2))
```


    
![png](SEULGI_story_to_PCA_files/SEULGI_story_to_PCA_11_0.png)
    





```
for i in tqdm(range(kmeans.cluster_centers_.shape[0])):
    data['center_angle'+str(i+1)] = 0
    for j in range(len(data)):
        data.loc[j, 'center_angle'+str(i+1)] =  cosim(kmeans.cluster_centers_[i,:], data.iloc[j, 1:kmeans.cluster_centers_.shape[1]+1].values )
```


    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))


    
    


```
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
      <th>df_index</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.027440</td>
      <td>-0.046735</td>
      <td>0.001222</td>
      <td>-0.007726</td>
      <td>-0.030526</td>
      <td>2</td>
      <td>0.459916</td>
      <td>-0.634579</td>
      <td>0.419405</td>
      <td>-0.312549</td>
      <td>-0.537552</td>
      <td>0.628538</td>
      <td>0.342371</td>
      <td>-0.183696</td>
      <td>0.529557</td>
      <td>-0.335597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>-0.139023</td>
      <td>-0.041158</td>
      <td>0.185709</td>
      <td>0.056368</td>
      <td>0.221978</td>
      <td>7</td>
      <td>0.115653</td>
      <td>0.305180</td>
      <td>0.265281</td>
      <td>-0.743291</td>
      <td>-0.283207</td>
      <td>-0.140113</td>
      <td>0.441106</td>
      <td>0.831863</td>
      <td>0.237650</td>
      <td>-0.269267</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.176497</td>
      <td>0.010671</td>
      <td>-0.104957</td>
      <td>0.049618</td>
      <td>-0.072200</td>
      <td>9</td>
      <td>-0.197856</td>
      <td>-0.363120</td>
      <td>-0.412866</td>
      <td>0.900482</td>
      <td>-0.021829</td>
      <td>-0.156486</td>
      <td>-0.543537</td>
      <td>-0.533412</td>
      <td>-0.308606</td>
      <td>0.766878</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>-0.007019</td>
      <td>-0.076433</td>
      <td>0.020901</td>
      <td>-0.046120</td>
      <td>-0.065877</td>
      <td>2</td>
      <td>0.653056</td>
      <td>-0.755046</td>
      <td>0.614338</td>
      <td>-0.071782</td>
      <td>-0.369635</td>
      <td>0.623845</td>
      <td>-0.149121</td>
      <td>-0.088530</td>
      <td>0.043266</td>
      <td>-0.177011</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>-0.061421</td>
      <td>-0.005673</td>
      <td>0.011988</td>
      <td>0.063355</td>
      <td>0.043213</td>
      <td>6</td>
      <td>-0.167815</td>
      <td>0.379185</td>
      <td>-0.130889</td>
      <td>-0.541992</td>
      <td>-0.388214</td>
      <td>-0.278352</td>
      <td>0.865509</td>
      <td>0.282917</td>
      <td>0.741507</td>
      <td>-0.270110</td>
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
    </tr>
    <tr>
      <th>2671</th>
      <td>5813</td>
      <td>0.079645</td>
      <td>0.105129</td>
      <td>-0.008806</td>
      <td>-0.044701</td>
      <td>0.081031</td>
      <td>4</td>
      <td>-0.450038</td>
      <td>0.492060</td>
      <td>-0.354798</td>
      <td>0.213064</td>
      <td>0.674289</td>
      <td>-0.312869</td>
      <td>-0.574690</td>
      <td>0.228302</td>
      <td>-0.736761</td>
      <td>0.315879</td>
    </tr>
    <tr>
      <th>2672</th>
      <td>5815</td>
      <td>0.200239</td>
      <td>-0.014906</td>
      <td>-0.010474</td>
      <td>0.087478</td>
      <td>0.077280</td>
      <td>9</td>
      <td>-0.164838</td>
      <td>-0.251193</td>
      <td>-0.303467</td>
      <td>0.552938</td>
      <td>-0.278057</td>
      <td>-0.338904</td>
      <td>-0.523017</td>
      <td>0.213205</td>
      <td>-0.394525</td>
      <td>0.944570</td>
    </tr>
    <tr>
      <th>2673</th>
      <td>5817</td>
      <td>-0.053887</td>
      <td>-0.066665</td>
      <td>-0.000806</td>
      <td>-0.072515</td>
      <td>-0.010311</td>
      <td>5</td>
      <td>0.211638</td>
      <td>-0.429370</td>
      <td>0.330430</td>
      <td>-0.647897</td>
      <td>-0.133609</td>
      <td>0.883245</td>
      <td>0.137083</td>
      <td>0.076858</td>
      <td>0.219064</td>
      <td>-0.442168</td>
    </tr>
    <tr>
      <th>2674</th>
      <td>5819</td>
      <td>-0.060442</td>
      <td>-0.024307</td>
      <td>-0.000970</td>
      <td>0.138879</td>
      <td>-0.008729</td>
      <td>6</td>
      <td>0.011169</td>
      <td>0.084262</td>
      <td>-0.111899</td>
      <td>-0.050660</td>
      <td>-0.592710</td>
      <td>-0.387972</td>
      <td>0.791635</td>
      <td>-0.125892</td>
      <td>0.816460</td>
      <td>-0.068338</td>
    </tr>
    <tr>
      <th>2675</th>
      <td>5823</td>
      <td>-0.043530</td>
      <td>0.028468</td>
      <td>0.054175</td>
      <td>0.055816</td>
      <td>0.007334</td>
      <td>1</td>
      <td>0.293310</td>
      <td>0.572018</td>
      <td>0.322213</td>
      <td>-0.222262</td>
      <td>-0.143784</td>
      <td>-0.696295</td>
      <td>0.609523</td>
      <td>0.326104</td>
      <td>0.360772</td>
      <td>-0.456063</td>
    </tr>
  </tbody>
</table>
<p>2676 rows × 17 columns</p>
</div>




```
df_made = pd.DataFrame(index=df.index).join(data.set_index('df_index'))
df_made.fillna(0, inplace=True)
df_made
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.027440</td>
      <td>-0.046735</td>
      <td>0.001222</td>
      <td>-0.007726</td>
      <td>-0.030526</td>
      <td>2.0</td>
      <td>0.459916</td>
      <td>-0.634579</td>
      <td>0.419405</td>
      <td>-0.312549</td>
      <td>-0.537552</td>
      <td>0.628538</td>
      <td>0.342371</td>
      <td>-0.183696</td>
      <td>0.529557</td>
      <td>-0.335597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.139023</td>
      <td>-0.041158</td>
      <td>0.185709</td>
      <td>0.056368</td>
      <td>0.221978</td>
      <td>7.0</td>
      <td>0.115653</td>
      <td>0.305180</td>
      <td>0.265281</td>
      <td>-0.743291</td>
      <td>-0.283207</td>
      <td>-0.140113</td>
      <td>0.441106</td>
      <td>0.831863</td>
      <td>0.237650</td>
      <td>-0.269267</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>5820</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5821</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5822</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5823</th>
      <td>-0.043530</td>
      <td>0.028468</td>
      <td>0.054175</td>
      <td>0.055816</td>
      <td>0.007334</td>
      <td>1.0</td>
      <td>0.293310</td>
      <td>0.572018</td>
      <td>0.322213</td>
      <td>-0.222262</td>
      <td>-0.143784</td>
      <td>-0.696295</td>
      <td>0.609523</td>
      <td>0.326104</td>
      <td>0.360772</td>
      <td>-0.456063</td>
    </tr>
    <tr>
      <th>5824</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5825 rows × 16 columns</p>
</div>




```
df_made.to_excel("스토리수치화.xlsx", index=False)
```


```
df = pd.read_excel("/content/최종일까3.xlsx")
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
      <th>index</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
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
      <td>0.512771</td>
      <td>0.052921</td>
      <td>0.034673</td>
      <td>0.288320</td>
      <td>0.111315</td>
      <td>0.152747</td>
      <td>0.188062</td>
      <td>0.189990</td>
      <td>0.164626</td>
      <td>0.128080</td>
      <td>0.062790</td>
      <td>0.049102</td>
      <td>0.054944</td>
      <td>0.056879</td>
      <td>0.086326</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
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
      <td>0.628889</td>
      <td>0.014945</td>
      <td>0.013700</td>
      <td>0.260274</td>
      <td>0.082193</td>
      <td>0.143124</td>
      <td>0.184568</td>
      <td>0.232524</td>
      <td>0.140583</td>
      <td>0.122165</td>
      <td>0.096614</td>
      <td>0.189901</td>
      <td>0.046541</td>
      <td>0.102961</td>
      <td>0.142104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
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
      <td>0.603444</td>
      <td>0.000002</td>
      <td>0.028737</td>
      <td>0.216475</td>
      <td>0.151341</td>
      <td>0.127590</td>
      <td>1.000000</td>
      <td>0.177255</td>
      <td>0.134459</td>
      <td>0.125366</td>
      <td>0.068866</td>
      <td>1.000000</td>
      <td>0.040797</td>
      <td>0.068504</td>
      <td>0.077417</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>3485</th>
      <td>5780</td>
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
      <td>0.170</td>
      <td>0.148</td>
      <td>0.682</td>
      <td>-0.9996</td>
      <td>0.538040</td>
      <td>0.028987</td>
      <td>0.043480</td>
      <td>0.271738</td>
      <td>0.117754</td>
      <td>0.121259</td>
      <td>0.190051</td>
      <td>0.080830</td>
      <td>0.124893</td>
      <td>0.118967</td>
      <td>0.123747</td>
      <td>0.098431</td>
      <td>0.141827</td>
      <td>0.122979</td>
      <td>0.112383</td>
    </tr>
    <tr>
      <th>3486</th>
      <td>5783</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3487</th>
      <td>5788</td>
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
      <td>0.118</td>
      <td>0.223</td>
      <td>0.659</td>
      <td>1.0000</td>
      <td>0.533232</td>
      <td>0.016250</td>
      <td>0.033975</td>
      <td>0.258493</td>
      <td>0.158051</td>
      <td>0.187390</td>
      <td>0.260704</td>
      <td>0.163689</td>
      <td>0.191616</td>
      <td>0.173335</td>
      <td>0.090000</td>
      <td>0.085901</td>
      <td>0.101430</td>
      <td>0.082965</td>
      <td>0.100139</td>
    </tr>
    <tr>
      <th>3488</th>
      <td>5807</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3489</th>
      <td>5823</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>3490 rows × 57 columns</p>
</div>




```
df = df.set_index("index")
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
      <td>0.512771</td>
      <td>0.052921</td>
      <td>0.034673</td>
      <td>0.288320</td>
      <td>0.111315</td>
      <td>0.152747</td>
      <td>0.188062</td>
      <td>0.189990</td>
      <td>0.164626</td>
      <td>0.128080</td>
      <td>0.062790</td>
      <td>0.049102</td>
      <td>0.054944</td>
      <td>0.056879</td>
      <td>0.086326</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>0.628889</td>
      <td>0.014945</td>
      <td>0.013700</td>
      <td>0.260274</td>
      <td>0.082193</td>
      <td>0.143124</td>
      <td>0.184568</td>
      <td>0.232524</td>
      <td>0.140583</td>
      <td>0.122165</td>
      <td>0.096614</td>
      <td>0.189901</td>
      <td>0.046541</td>
      <td>0.102961</td>
      <td>0.142104</td>
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
      <td>0.603444</td>
      <td>0.000002</td>
      <td>0.028737</td>
      <td>0.216475</td>
      <td>0.151341</td>
      <td>0.127590</td>
      <td>1.000000</td>
      <td>0.177255</td>
      <td>0.134459</td>
      <td>0.125366</td>
      <td>0.068866</td>
      <td>1.000000</td>
      <td>0.040797</td>
      <td>0.068504</td>
      <td>0.077417</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>0.170</td>
      <td>0.148</td>
      <td>0.682</td>
      <td>-0.9996</td>
      <td>0.538040</td>
      <td>0.028987</td>
      <td>0.043480</td>
      <td>0.271738</td>
      <td>0.117754</td>
      <td>0.121259</td>
      <td>0.190051</td>
      <td>0.080830</td>
      <td>0.124893</td>
      <td>0.118967</td>
      <td>0.123747</td>
      <td>0.098431</td>
      <td>0.141827</td>
      <td>0.122979</td>
      <td>0.112383</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>0.118</td>
      <td>0.223</td>
      <td>0.659</td>
      <td>1.0000</td>
      <td>0.533232</td>
      <td>0.016250</td>
      <td>0.033975</td>
      <td>0.258493</td>
      <td>0.158051</td>
      <td>0.187390</td>
      <td>0.260704</td>
      <td>0.163689</td>
      <td>0.191616</td>
      <td>0.173335</td>
      <td>0.090000</td>
      <td>0.085901</td>
      <td>0.101430</td>
      <td>0.082965</td>
      <td>0.100139</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>3490 rows × 56 columns</p>
</div>




```
df = df.join(df_made, how='left')
```


```
df = df.reset_index()
df
#df.to_excel("최종일까4.xlsx")
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
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
      <td>0.512771</td>
      <td>0.052921</td>
      <td>0.034673</td>
      <td>0.288320</td>
      <td>0.111315</td>
      <td>0.152747</td>
      <td>0.188062</td>
      <td>0.189990</td>
      <td>0.164626</td>
      <td>0.128080</td>
      <td>0.062790</td>
      <td>0.049102</td>
      <td>0.054944</td>
      <td>0.056879</td>
      <td>0.086326</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.007019</td>
      <td>-0.076433</td>
      <td>0.020901</td>
      <td>-0.046120</td>
      <td>-0.065877</td>
      <td>2.0</td>
      <td>0.653056</td>
      <td>-0.755046</td>
      <td>0.614338</td>
      <td>-0.071782</td>
      <td>-0.369635</td>
      <td>0.623845</td>
      <td>-0.149121</td>
      <td>-0.088530</td>
      <td>0.043266</td>
      <td>-0.177011</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
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
      <td>0.628889</td>
      <td>0.014945</td>
      <td>0.013700</td>
      <td>0.260274</td>
      <td>0.082193</td>
      <td>0.143124</td>
      <td>0.184568</td>
      <td>0.232524</td>
      <td>0.140583</td>
      <td>0.122165</td>
      <td>0.096614</td>
      <td>0.189901</td>
      <td>0.046541</td>
      <td>0.102961</td>
      <td>0.142104</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
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
      <td>0.603444</td>
      <td>0.000002</td>
      <td>0.028737</td>
      <td>0.216475</td>
      <td>0.151341</td>
      <td>0.127590</td>
      <td>1.000000</td>
      <td>0.177255</td>
      <td>0.134459</td>
      <td>0.125366</td>
      <td>0.068866</td>
      <td>1.000000</td>
      <td>0.040797</td>
      <td>0.068504</td>
      <td>0.077417</td>
      <td>0.046832</td>
      <td>0.050008</td>
      <td>0.000827</td>
      <td>-0.056246</td>
      <td>-0.000731</td>
      <td>2.0</td>
      <td>-0.093938</td>
      <td>0.229441</td>
      <td>-0.015535</td>
      <td>0.372829</td>
      <td>0.783750</td>
      <td>-0.074603</td>
      <td>-0.812511</td>
      <td>-0.007069</td>
      <td>-0.913350</td>
      <td>0.130786</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
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
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.010356</td>
      <td>-0.049705</td>
      <td>-0.044241</td>
      <td>-0.044423</td>
      <td>0.042522</td>
      <td>5.0</td>
      <td>-0.464007</td>
      <td>-0.396039</td>
      <td>-0.369689</td>
      <td>-0.541423</td>
      <td>-0.115446</td>
      <td>0.877661</td>
      <td>-0.016348</td>
      <td>0.095757</td>
      <td>0.140356</td>
      <td>0.165168</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>3485</th>
      <td>5780</td>
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
      <td>0.170</td>
      <td>0.148</td>
      <td>0.682</td>
      <td>-0.9996</td>
      <td>0.538040</td>
      <td>0.028987</td>
      <td>0.043480</td>
      <td>0.271738</td>
      <td>0.117754</td>
      <td>0.121259</td>
      <td>0.190051</td>
      <td>0.080830</td>
      <td>0.124893</td>
      <td>0.118967</td>
      <td>0.123747</td>
      <td>0.098431</td>
      <td>0.141827</td>
      <td>0.122979</td>
      <td>0.112383</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3486</th>
      <td>5783</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043667</td>
      <td>0.107572</td>
      <td>0.022634</td>
      <td>0.000226</td>
      <td>-0.005313</td>
      <td>1.0</td>
      <td>-0.059462</td>
      <td>0.660612</td>
      <td>-0.038620</td>
      <td>0.520412</td>
      <td>0.709809</td>
      <td>-0.737231</td>
      <td>-0.412521</td>
      <td>-0.035005</td>
      <td>-0.639825</td>
      <td>0.013414</td>
    </tr>
    <tr>
      <th>3487</th>
      <td>5788</td>
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
      <td>0.118</td>
      <td>0.223</td>
      <td>0.659</td>
      <td>1.0000</td>
      <td>0.533232</td>
      <td>0.016250</td>
      <td>0.033975</td>
      <td>0.258493</td>
      <td>0.158051</td>
      <td>0.187390</td>
      <td>0.260704</td>
      <td>0.163689</td>
      <td>0.191616</td>
      <td>0.173335</td>
      <td>0.090000</td>
      <td>0.085901</td>
      <td>0.101430</td>
      <td>0.082965</td>
      <td>0.100139</td>
      <td>0.045578</td>
      <td>0.192430</td>
      <td>-0.051010</td>
      <td>-0.003610</td>
      <td>0.038821</td>
      <td>4.0</td>
      <td>-0.542302</td>
      <td>0.755464</td>
      <td>-0.493402</td>
      <td>0.356683</td>
      <td>0.810768</td>
      <td>-0.536391</td>
      <td>-0.233908</td>
      <td>-0.209098</td>
      <td>-0.431474</td>
      <td>0.052205</td>
    </tr>
    <tr>
      <th>3488</th>
      <td>5807</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.122976</td>
      <td>-0.015531</td>
      <td>-0.056793</td>
      <td>0.089688</td>
      <td>0.019662</td>
      <td>9.0</td>
      <td>-0.301822</td>
      <td>-0.304320</td>
      <td>-0.506487</td>
      <td>0.667796</td>
      <td>-0.335329</td>
      <td>-0.295477</td>
      <td>-0.322057</td>
      <td>-0.161646</td>
      <td>-0.115399</td>
      <td>0.929672</td>
    </tr>
    <tr>
      <th>3489</th>
      <td>5823</td>
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
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.043530</td>
      <td>0.028468</td>
      <td>0.054175</td>
      <td>0.055816</td>
      <td>0.007334</td>
      <td>1.0</td>
      <td>0.293310</td>
      <td>0.572018</td>
      <td>0.322213</td>
      <td>-0.222262</td>
      <td>-0.143784</td>
      <td>-0.696295</td>
      <td>0.609523</td>
      <td>0.326104</td>
      <td>0.360772</td>
      <td>-0.456063</td>
    </tr>
  </tbody>
</table>
<p>3490 rows × 73 columns</p>
</div>




```
df.to_excel("최종일까4.xlsx", index=False)
```


```
import imb
```
