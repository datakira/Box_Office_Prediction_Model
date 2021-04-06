```python
!pip install selenium
!pip install webdriver-manager
```

    Requirement already satisfied: selenium in c:\users\beomj\anaconda3\lib\site-packages (3.141.0)
    Requirement already satisfied: urllib3 in c:\users\beomj\anaconda3\lib\site-packages (from selenium) (1.25.9)
    Requirement already satisfied: webdriver-manager in c:\users\beomj\anaconda3\lib\site-packages (3.2.1)
    Requirement already satisfied: configparser in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (5.0.0)
    Requirement already satisfied: requests in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (2.24.0)
    Requirement already satisfied: crayons in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (0.3.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (1.25.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (3.0.4)
    Requirement already satisfied: colorama in c:\users\beomj\anaconda3\lib\site-packages (from crayons->webdriver-manager) (0.4.3)
    


```python
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
from tqdm import tqdm
import numpy as np
```


```python
df = pd.read_excel("1970_2019_영화.xlsx")
names = df.movieNm
years = df.prdtYear
```


```python
addr = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=149236" 

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(addr)
```

    [WDM] - Current google-chrome version is 84.0.4147
    [WDM] - Get LATEST driver version for 84.0.4147
    [WDM] - Driver [C:\Users\beomj\.wdm\drivers\chromedriver\win32\84.0.4147.30\chromedriver.exe] found in cache
     
    


```python
def find(name, year):
    stories = []
    rates = []
    reviews = []
    try:
        temp = driver.find_element_by_class_name("ipt_tx_srch")
        temp.send_keys(name)
        driver.find_element_by_class_name("btn_srch").click()
        time.sleep(1)
        temp = driver.find_element_by_class_name("search_list_1")
        time.sleep(1)
        temp = temp.find_elements_by_tag_name("li")
        found =False
        for i in temp:
            if year in i.text:
                found =True
                temp = i.find_element_by_tag_name("dt")
                temp = temp.find_element_by_tag_name("a").click()
                temp = driver.find_element_by_class_name("story_area")
                break
        if not found:
            for i in temp:
                if str(int(year)-1) in i.text or str(int(year)+1) in i.text :
                    found =True
                    temp = i.find_element_by_tag_name("dt")
                    temp = temp.find_element_by_tag_name("a").click()
                    temp = driver.find_element_by_class_name("story_area")
                    break
        stories.append(temp.text)

        try:
            temp = driver.find_element_by_id("movieEndTabMenu")
            temp.find_element_by_link_text("평점").click()
            time.sleep(0.5)
            temp = driver.find_element_by_id("netizen_point_tab_inner")
            rates.append(temp.text)
        except:
            rates.append("")    
        
        try:
            total = []
            for i in range(1, 50):
                driver.switch_to_default_content()
                driver.switch_to_frame('pointAfterListIframe')
                try:
                    if i != 1:
                        driver.find_element_by_id("pagerTagAnchor"+str(i)).click()
                        time.sleep(0.3)
                except:
                    break
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                content_list = soup.find("div", class_ = "score_result").find_all('li')
                try:
                    for li in content_list :
                        temp_review = li.find("div", class_ = "score_reple").find("p").get_text()
                        total.append(temp_review.strip())
                except:
                    pass
            total = "///".join(total)
            total = total.replace("\n", "")
            total = total.replace("\t", "")
            total = total.replace("관람객", "")
            total = total.replace("스포일러가 포함된 감상평입니다. 감상평 보기", "")
            reviews.append(total)
        except:
            reviews.append("")
    except:
        stories.append("")
        rates.append("")
        reviews.append("")
        
        addr = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=149236" 
        driver.get(addr)
    addr = "https://movie.naver.com/movie/bi/mi/basic.nhn?code=149236" 
    driver.get(addr)
    return stories, rates, reviews
```


```python
data = pd.DataFrame({'영화명':names})
data['스토리'] = None
data['평점'] = None
data['리뷰'] = None
data.head()
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
      <th>영화명</th>
      <th>스토리</th>
      <th>평점</th>
      <th>리뷰</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>지니어스 독</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>뉴 뮤턴트</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>드라이브</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>쥬라기 썬더</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>바다 몬스터2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
start = 4000
end = len(df)
for i in tqdm(range(start, end)):
    if data.loc[i, '스토리'] is None:
        try:
            year = str(int(years[i]))
        except:
            year = 0
        name = names[i]
        s,r,re = find(name, year)
        data.loc[i,'스토리'] =s[0]  
        data.loc[i, '평점']  =r[0]  
        data.loc[i, '리뷰']  =re[0]  
```

    100%|██████████| 1825/1825 [7:39:08<00:00, 15.10s/it]
    


```python
data.to_excel("1970_2019_naver"+str(start)+"_"+str(end)+".xlsx", index=False)
```


```python
data[4000:]
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
      <th>영화명</th>
      <th>스토리</th>
      <th>평점</th>
      <th>리뷰</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4000</th>
      <td>블라이가의 미로</td>
      <td>줄거리\n대학을 졸업하고 직장을 찾던 제니(Jenny: 팻시 켄싯 분)는 쿠퍼(Mr...</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4001</th>
      <td>노 임팩트 맨</td>
      <td>줄거리\n작가이자 환경운동가인 ‘콜린’은 1년간 가족과 함께 지구에 무해(無害)한 ...</td>
      <td>8.72</td>
      <td>노 임팩트맨 책을 읽고 보는데 저희에게 큰 임팩트를 주는 것 같아요. 정치인을 좋은...</td>
    </tr>
    <tr>
      <th>4002</th>
      <td>오페라- 햄릿 (메트로폴리탄)</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4003</th>
      <td>A-특공대</td>
      <td>줄거리\n최고의 실력을 자랑하던 특공대가 돌연 자취를 감춘 지 1년. 누구도 해결할...</td>
      <td>8.93</td>
      <td>이도 저도 아닌 어중간한 영화를 만들바엔 차라리 이렇게 만들어야한다///기승전결이 ...</td>
    </tr>
    <tr>
      <th>4004</th>
      <td>섹스 앤 더 시티 2</td>
      <td>줄거리\n더 화려하게 더 당당하게\n캐리(사라 제시카 파커)가 오랜 연인이었던 빅과...</td>
      <td>7.75</td>
      <td>된장녀라 사치녀라 불려도 내겐 부럽고 멋있는 여성들의 스토리를 담은 최고의 드라마/...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5820</th>
      <td>한나스 워</td>
      <td>줄거리\n1921년 부다페스트에서 저명한 극작가의 딸로 태어난 한나(Hanna: 마...</td>
      <td>9.39</td>
      <td>좀 오래된 영화이지만 명작하라 하기엔 충분하다///내가 본 최고의 명화이지만 찾아보...</td>
    </tr>
    <tr>
      <th>5821</th>
      <td>핫스 오브 화이어</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5822</th>
      <td>핫타켓</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5823</th>
      <td>햄버거 힐</td>
      <td>줄거리\n{제101공수부대(Troops Of The 101st Airborne Di...</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5824</th>
      <td>헐크호건의 죽느냐 사느냐</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>1825 rows × 4 columns</p>
</div>




```python

```
