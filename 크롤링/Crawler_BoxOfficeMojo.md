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
    Requirement already satisfied: idna<3,>=2.5 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (1.25.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2020.6.20)
    


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
addr = "https://www.boxofficemojo.com/search/?q=Think+Like+a+Dog" 

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(addr)

stories = []
a_release = []
o_release = []
```

    [WDM] - Current google-chrome version is 84.0.4147
    [WDM] - Get LATEST driver version for 84.0.4147
    [WDM] - Driver [C:\Users\beomj\.wdm\drivers\chromedriver\win32\84.0.4147.30\chromedriver.exe] found in cache
     
    


```python
df2 = pd.read_excel("1970_2019_영화.xlsx")
df2.prdtYear.fillna(0, inplace=True)
years = df2.prdtYear.astype(int).astype(str)
names = df2.movieNmEn	
```


```python
df = pd.DataFrame({"영화명EN": names})
df['스토리']=None
df['A_release']=None
df['O_release']=None
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
      <th>영화명EN</th>
      <th>스토리</th>
      <th>A_release</th>
      <th>O_release</th>
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5820</th>
      <td>Hanna'S War</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5821</th>
      <td>Hearts Of Fire</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5822</th>
      <td>Hot Target</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5823</th>
      <td>Hamberger Hill</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5824</th>
      <td>No Holds Barred</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>5825 rows × 4 columns</p>
</div>




```python
def find(name, year):
    stories = []
    a_release = []
    o_release = []
    try:
        driver.find_element_by_id("mojo-search-text-input").send_keys(name)
        driver.find_element_by_id("mojo-search-button").click()
        while True:
            try:
                time.sleep(1)
                driver.find_element_by_id("body")
                break
            except:
                pass
        temp = driver.find_element_by_xpath("/html/body/div[1]/main/div/div/div")
        temp = temp.find_elements_by_class_name("a-fixed-left-grid-inner")
        found = False
        for i in temp:
            if year in i.text:
                i.find_element_by_tag_name("a").click()
                found =True
                break
        if not found:
            for i in temp:
                if str(int(year)-1) in i.text or str(int(year)+1) in i.text:
                    i.find_element_by_tag_name("a").click()
                    found =True
                    break
        if not found:
            stories.append("")
            a_release.append("")
            o_release.append("")
            return stories, a_release, o_release
        #-----------------------
        try:
            stories.append(driver.find_element_by_class_name("mojo-heading-summary").text)
        except:
            stories.append("")
        #-----------------------
        try:
            a_release.append(driver.find_element_by_class_name("mojo-summary-table").text)
        except:
            a_release.append("")
        #------------------------
        try:
            temp = driver.find_element_by_class_name("a-dropdown-prompt").click()
            driver.find_element_by_link_text("Original Release").click()
            while True:
                try:
                    time.sleep(0.7)
                    total = []
                    table = driver.find_elements_by_class_name("mojo-table")
                    for i in table:
                        total.append(i.text)
                    o_release.append("\n".join(total))
                    if table:
                        break
                except:
                    pass
        except:
            o_release.append("")
        #-----------------------
    except:
        stories.append("")
        a_release.append("")
        o_release.append("")
    driver.get(addr)
    return stories, a_release, o_release
```


```python
start = 4800
end = len(df)
for i in tqdm(range(start, end)):
    name = names[i]
    year = years[i]
    s, a, o = find(name, year) 
    df.loc[i, '스토리'] = s[0]
    df.loc[i, 'A_release'] = a[0]
    df.loc[i, 'O_release'] = o[0]        
    
```

    100%|██████████| 1025/1025 [1:39:29<00:00,  5.82s/it]
    


```python
df.to_excel("1970_2019_mojo"+str(start)+"_"+str(end)+".xlsx", index=False)
```


```python
df[4800:].A_release.isna().sum()
```




    0




```python
(df[4800:].A_release=="").sum()
```




    258




```python
(df[4800:].A_release=="")
```
