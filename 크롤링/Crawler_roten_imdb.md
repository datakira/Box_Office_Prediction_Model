```python
!pip install selenium
!pip install webdriver-manager
```

    Requirement already satisfied: selenium in c:\users\beomj\anaconda3\lib\site-packages (3.141.0)
    Requirement already satisfied: urllib3 in c:\users\beomj\anaconda3\lib\site-packages (from selenium) (1.25.9)
    Requirement already satisfied: webdriver-manager in c:\users\beomj\anaconda3\lib\site-packages (3.2.1)
    Requirement already satisfied: configparser in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (5.0.0)
    Requirement already satisfied: crayons in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (0.3.1)
    Requirement already satisfied: requests in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (2.24.0)
    Requirement already satisfied: colorama in c:\users\beomj\anaconda3\lib\site-packages (from crayons->webdriver-manager) (0.4.3)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (1.25.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2.10)
    


```python
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions

from tqdm import tqdm
```


```python
addr = "https://www.imdb.com/find?q=deadpool&ref_=nv_sr_sm" 

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(addr)

```

    [WDM] - Current google-chrome version is 84.0.4147
    [WDM] - Get LATEST driver version for 84.0.4147
    [WDM] - Driver [C:\Users\beomj\.wdm\drivers\chromedriver\win32\84.0.4147.30\chromedriver.exe] found in cache
     
    


```python
df = pd.read_excel("영화raw데이터.xlsx")
df.openDt.fillna(0, inplace=True)
years = df.openDt.apply(lambda x:str(x)[:4])
```


```python
indexes = df[df.kor_revenue.notnull()].index.values
print(indexes)
df.head()
```

    [   2    4    7 ... 5788 5807 5823]
    




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
      <th>actors</th>
      <th>...</th>
      <th>companys</th>
      <th>Domestic</th>
      <th>Budget</th>
      <th>Distributor</th>
      <th>review</th>
      <th>MPAA</th>
      <th>raters</th>
      <th>ratings</th>
      <th>kor_revenue</th>
      <th>kor_audience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20203303</td>
      <td>지니어스 독</td>
      <td>Think Like a Dog</td>
      <td>90.0</td>
      <td>2020.0</td>
      <td>20200916</td>
      <td>개봉예정</td>
      <td>코미디,드라마</td>
      <td>길 정거</td>
      <td>메간 폭스,조쉬 더하멜,가브리엘 베이트먼</td>
      <td>...</td>
      <td>20168728,(주)키다리이엔티,배급사,20153568,(주)스톰픽쳐스코리아,수입사</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>667</td>
      <td>5.1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20175405</td>
      <td>뉴 뮤턴트</td>
      <td>The New Mutants</td>
      <td>98.0</td>
      <td>2018.0</td>
      <td>20200903</td>
      <td>개봉예정</td>
      <td>액션,공포(호러),SF</td>
      <td>조쉬 분</td>
      <td>메이지 윌리암스,안야 테일러 조이,앨리스 브라가</td>
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
      <th>2</th>
      <td>20112692</td>
      <td>드라이브</td>
      <td>Drive</td>
      <td>100.0</td>
      <td>2011.0</td>
      <td>20111117</td>
      <td>개봉</td>
      <td>액션,멜로/로맨스</td>
      <td>니콜라스 윈딩 레픈</td>
      <td>라이언 고슬링,캐리 멀리건</td>
      <td>...</td>
      <td>20111613,주식회사 풍경소리,배급사,20100285,판씨네마(주),배급사,20...</td>
      <td>$35,061,555</td>
      <td>$15,000,000</td>
      <td>FilmDistrict</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>156237980.0</td>
      <td>20141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20205511</td>
      <td>쥬라기 썬더</td>
      <td>Jurassic Thunder</td>
      <td>84.0</td>
      <td>2019.0</td>
      <td>20200923</td>
      <td>개봉예정</td>
      <td>액션,SF</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>....but then again, did you expect an award wi...</td>
      <td>15</td>
      <td>142</td>
      <td>1.9</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20203043</td>
      <td>바다 몬스터2</td>
      <td>Sea Monsters2</td>
      <td>87.0</td>
      <td>2018.0</td>
      <td>20200812</td>
      <td>개봉</td>
      <td>애니메이션</td>
      <td>에반 트라멜</td>
      <td>서반석,석승훈</td>
      <td>...</td>
      <td>20203701,(주)블루라벨픽쳐스,배급사,20190541,(유)헤이데이웍스,수입사</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Loose plot with no content. The story is vauge...</td>
      <td>NaN</td>
      <td>30</td>
      <td>3.7</td>
      <td>5000.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df2 = df[['movieNmEn']]
df2['평점_en'] = None
df2['리뷰_en'] = None
df2['시청등급_en'] = None
df2.head()
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
      <th>movieNmEn</th>
      <th>평점_en</th>
      <th>리뷰_en</th>
      <th>시청등급_en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Think Like a Dog</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Mutants</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drive</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jurassic Thunder</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sea Monsters2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
def find(name, year):
    info = []
    reviews = []
    rates = []
    try:
        driver.find_element_by_xpath("/html/body/div[2]/nav/div[2]/div[1]/form/div[2]/div/input").send_keys(name)
        driver.find_element_by_id("suggestion-search-button").click()      
        time.sleep(1.5)
        #------------------
        temp = driver.find_element_by_class_name("findList")
        table = temp.find_elements_by_class_name('result_text')
        found = False
        for temp in table:
            text = temp.text.lower()
            name_split = name.lower().split()
            cnt = 0
            for i in name_split:
                if i in text:
                    cnt +=1
            #print(text, name_split, year)
            if cnt >= len(name_split)/2 and (year in text or str(int(year)-1) in text or str(int(year)+1) in text):
                temp = temp.find_element_by_tag_name("a")
                temp.click()
                time.sleep(1.5)
                found = True
                break

        if not found:
            reviews.append("")
            rates.append("")
            info.append("")
            driver.get(addr)
            return info, reviews, rates
        try:
            #------------------
            rates.append(driver.find_element_by_class_name("imdbRating").text)
            temp = driver.find_element_by_class_name("title_block")
            temp = temp.find_element_by_class_name("subtext").text
            info.append(temp.split("|")[0].strip())

            driver.find_element_by_link_text("USER REVIEWS").click()
            try:
                for i in range(10):
                    driver.find_element_by_id("load-more-trigger").click()
                    time.sleep(2)
            except:
                pass
            time.sleep(1.5)
            #------------------
            temp = driver.find_element_by_class_name("lister-list")
            temp = temp.find_elements_by_class_name("text")
            total = []
            for i in temp:
                total.append(i.text)
            total = "///".join(total)
            reviews.append(total)
        except:
            reviews.append("")
    except:
        reviews.append("")
        rates.append("")
        info.append("")
        driver.get(addr)
    return info, reviews, rates

```


```python
start = 0
end = 2000
for i in tqdm([i for i in range(start, end) if i in indexes]):
    info, review, rate = find(df2.loc[i,'movieNmEn'], str(int(years[i])))
    df2.loc[i, '리뷰_en']  = review
    df2.loc[i, '평점_en']  = rate
    df2.loc[i, '시청등급_en'] = info

```

    100%|██████████| 1956/1956 [10:46:54<00:00, 19.84s/it]
    


```python
pd.set_option("display.max_rows", 30)
df2[start:end]
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
      <th>movieNmEn</th>
      <th>평점_en</th>
      <th>리뷰_en</th>
      <th>시청등급_en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Think Like a Dog</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Mutants</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drive</td>
      <td>[7.8/10\n558,449]</td>
      <td>[///One reviewer here suggested that instead o...</td>
      <td>[18]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jurassic Thunder</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sea Monsters2</td>
      <td>[]</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>Beyond the Lights</td>
      <td>[6.9/10\n15,079]</td>
      <td>[I had never heard of this movie before a coll...</td>
      <td>[15]</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>The Equalizer</td>
      <td>[7.2/10\n325,331]</td>
      <td>[Really great action sequences, with some nice...</td>
      <td>[18]</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>UNIVERSAL SQUADRONS</td>
      <td>[]</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>Lullaby</td>
      <td>[6.2/10\n2,528]</td>
      <td>[This is a wonderful spectacular and emotional...</td>
      <td>[15]</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>You're Not You</td>
      <td>[7.3/10\n23,703]</td>
      <td>[This movie brought me to tears.\n\nOur mother...</td>
      <td>[15]</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 4 columns</p>
</div>




```python
df2.to_excel("1970_2019_review_en"+str(start)+"_"+str(end)+".xlsx", index=False)
```


```python

```
