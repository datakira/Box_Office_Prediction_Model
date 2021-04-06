```python
!pip install selenium
!pip install webdriver-manager
```

    Requirement already satisfied: selenium in c:\users\beomj\anaconda3\lib\site-packages (3.141.0)
    Requirement already satisfied: urllib3 in c:\users\beomj\anaconda3\lib\site-packages (from selenium) (1.25.9)
    Requirement already satisfied: webdriver-manager in c:\users\beomj\anaconda3\lib\site-packages (3.2.1)
    Requirement already satisfied: requests in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (2.24.0)
    Requirement already satisfied: configparser in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (5.0.0)
    Requirement already satisfied: crayons in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (0.3.1)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (1.25.9)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2020.6.20)
    Requirement already satisfied: colorama in c:\users\beomj\anaconda3\lib\site-packages (from crayons->webdriver-manager) (0.4.3)
    


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
df = pd.read_excel("1970_2020_raw영화데이터_kobiz_update.xlsx")
#df = df[['movieCd','movieNm', 'prdtYear']]
#df['stats'] = None
names = df.movieNm.astype(str)

df.head()
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
      <th>actors</th>
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
      <td>2D</td>
      <td>12세이상관람가</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12</td>
      <td>NaN</td>
      <td>667</td>
      <td>5.1</td>
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
      <td>2D</td>
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
      <td>필름,2D</td>
      <td>청소년관람불가,청소년관람불가</td>
      <td>$35,061,555</td>
      <td>$15,000,000</td>
      <td>FilmDistrict</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>144975500.0</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>2D</td>
      <td>15세이상관람가</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>....but then again, did you expect an award wi...</td>
      <td>15</td>
      <td>NaN</td>
      <td>142</td>
      <td>1.9</td>
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
      <td>2D,2D</td>
      <td>전체관람가</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Loose plot with no content. The story is vauge...</td>
      <td>NaN</td>
      <td>5000.0</td>
      <td>30</td>
      <td>3.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df[df.Domestic.notnull()][df.Domestic !="-"]
indexes = df2.index.values
len(indexes)
```




    3702




```python
df['stats2'] = 0
```


```python
addr = "http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do" 
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(addr)
```

    [WDM] - Current google-chrome version is 84.0.4147
    [WDM] - Get LATEST driver version for 84.0.4147
    [WDM] - Driver [C:\Users\beomj\.wdm\drivers\chromedriver\win32\84.0.4147.30\chromedriver.exe] found in cache
     
    


    ---------------------------------------------------------------------------

    SessionNotCreatedException                Traceback (most recent call last)

    <ipython-input-39-345993bb6501> in <module>
          1 addr = "http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do"
    ----> 2 driver = webdriver.Chrome(ChromeDriverManager().install())
          3 driver.get(addr)
    

    ~\Anaconda3\lib\site-packages\selenium\webdriver\chrome\webdriver.py in __init__(self, executable_path, port, options, service_args, desired_capabilities, service_log_path, chrome_options, keep_alive)
         79                     remote_server_addr=self.service.service_url,
         80                     keep_alive=keep_alive),
    ---> 81                 desired_capabilities=desired_capabilities)
         82         except Exception:
         83             self.quit()
    

    ~\Anaconda3\lib\site-packages\selenium\webdriver\remote\webdriver.py in __init__(self, command_executor, desired_capabilities, browser_profile, proxy, keep_alive, file_detector, options)
        155             warnings.warn("Please use FirefoxOptions to set browser profile",
        156                           DeprecationWarning, stacklevel=2)
    --> 157         self.start_session(capabilities, browser_profile)
        158         self._switch_to = SwitchTo(self)
        159         self._mobile = Mobile(self)
    

    ~\Anaconda3\lib\site-packages\selenium\webdriver\remote\webdriver.py in start_session(self, capabilities, browser_profile)
        250         parameters = {"capabilities": w3c_caps,
        251                       "desiredCapabilities": capabilities}
    --> 252         response = self.execute(Command.NEW_SESSION, parameters)
        253         if 'sessionId' not in response:
        254             response = response['value']
    

    ~\Anaconda3\lib\site-packages\selenium\webdriver\remote\webdriver.py in execute(self, driver_command, params)
        319         response = self.command_executor.execute(driver_command, params)
        320         if response:
    --> 321             self.error_handler.check_response(response)
        322             response['value'] = self._unwrap_value(
        323                 response.get('value', None))
    

    ~\Anaconda3\lib\site-packages\selenium\webdriver\remote\errorhandler.py in check_response(self, response)
        240                 alert_text = value['alert'].get('text')
        241             raise exception_class(message, screen, stacktrace, alert_text)
    --> 242         raise exception_class(message, screen, stacktrace)
        243 
        244     def _value_or_default(self, obj, key, default):
    

    SessionNotCreatedException: Message: session not created
    from chrome not reachable
      (Session info: chrome=84.0.4147.135)
    



```python
def find(name, cd):
    driver.find_element_by_name("sMovName").clear()
    driver.find_element_by_name("sMovName").send_keys(name)
    driver.find_element_by_class_name("btn_sch").click()
    time.sleep(1)
    try:
        found =False
        for i in range(2, 11):
            temp = driver.find_element_by_class_name("rst_sch")
            temp = temp.find_elements_by_link_text(name)
            for t in temp:
                t.click()
                temp = driver.find_element_by_class_name("item_tab")
                text = temp.find_element_by_class_name("cont").text
                if cd not in text:
                    driver.find_element_by_link_text("뒤로").click()   
                else:
                    found = True
                    break
            if found:
                break
            temp = driver.find_element_by_id("pagingForm")
            temp.find_element_by_link_text(str(i)).click()
            time.sleep(0.5)
        if not found:
            return None
    except:
        return None

    temp = driver.find_element_by_class_name("wrap_tab")
    temp.find_element_by_link_text('통계정보').click()
    while True:
        try:
            time.sleep(2) 
            temp = driver.find_element_by_class_name("statistics")
            infos = temp.find_elements_by_class_name("info")
            if infos:
                break
        except:
            pass
    
    total = []
    #found = False
    for j in infos:
        if "KOBIS(발권)통계" in j.text:
            tables = j.find_elements_by_tag_name("table")
            for i in tables:
                if '전국' in i.text:
                    #found=True
                    total.append(i.text[i.text.find("전국"):])
            break
            #if found:
        #    break
    driver.find_element_by_link_text("뒤로").click()   
    #if not found:
    #    return None
    #total = " ".join(total).split('\n')
    total = " ".join(total)
    #for i in total:
    return total
```


```python
start = 61
end = 70
for i in tqdm(range(start, end)):
    if i in indexes:
   # if not df.loc[i, 'stats']:
        name = df.loc[i, 'movieNm'].strip()
        cd   = df.loc[i, 'movieCd']
        try:
            text = find(name, cd)
            df.loc[i, 'stats2'] = text
            print(i, text[:20])
        except:
            addr = "http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do" 
            driver.get(addr)
            df.loc[i, 'stats2'] = None
```

    11%|█         | 1/9 [00:07<01:00,  7.52s/it]61 전국 301 5,799,168,000
     22%|██▏       | 2/9 [00:15<00:53,  7.59s/it]62 전국 333 88,807,180 (1
     44%|████▍     | 4/9 [00:24<00:33,  6.73s/it]64 전국 258 12,661,780,40
     56%|█████▌    | 5/9 [00:33<00:29,  7.30s/it]65 전국 279 165,941,860 (
     89%|████████▉ | 8/9 [00:40<00:05,  5.85s/it]68 전국 19 20,213,500 (10
    100%|██████████| 9/9 [00:50<00:00,  5.56s/it]69 전국 156 264,874,260 (
    
    


```python
df.to_excel("1970_2019_코비스매출_stats2"+str(start)+"_"+str(end)+".xlsx", index=False)
```


```python
pd.set_option('display.max_rows', 80)
df.loc[indexes].loc[start:end][['movieNm','stats', 'stats2']]
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
      <th>movieNm</th>
      <th>stats</th>
      <th>stats2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>배트맨 비긴즈</td>
      <td>NaN</td>
      <td>전국 301 5,799,168,000 (100%) 903,190 (100%)</td>
    </tr>
    <tr>
      <th>62</th>
      <td>라스트 풀 메저</td>
      <td>88807180.0</td>
      <td>전국 333 88,807,180 (100%) 11,223 (100%)</td>
    </tr>
    <tr>
      <th>64</th>
      <td>색, 계</td>
      <td>0.0</td>
      <td>전국 258 12,661,780,400 (100%) 1,925,774 (100%)</td>
    </tr>
    <tr>
      <th>65</th>
      <td>잃어버린 세계를 찾아서</td>
      <td>161271360.0</td>
      <td>전국 279 165,941,860 (100%) 21,879 (100%)</td>
    </tr>
    <tr>
      <th>68</th>
      <td>리플리</td>
      <td>0.0</td>
      <td>전국 19 20,213,500 (100%) 2,605 (100%)</td>
    </tr>
    <tr>
      <th>69</th>
      <td>아이 캔 온리 이매진</td>
      <td>182789600.0</td>
      <td>전국 156 264,874,260 (100%) 34,859 (100%)</td>
    </tr>
    <tr>
      <th>70</th>
      <td>플라이</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


