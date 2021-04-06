```python
!pip install selenium
!pip install webdriver-manager
```

    Requirement already satisfied: selenium in c:\users\beomj\anaconda3\lib\site-packages (3.141.0)
    Requirement already satisfied: urllib3 in c:\users\beomj\anaconda3\lib\site-packages (from selenium) (1.25.9)
    Requirement already satisfied: webdriver-manager in c:\users\beomj\anaconda3\lib\site-packages (3.2.1)
    Requirement already satisfied: crayons in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (0.3.1)
    Requirement already satisfied: configparser in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (5.0.0)
    Requirement already satisfied: requests in c:\users\beomj\anaconda3\lib\site-packages (from webdriver-manager) (2.24.0)
    Requirement already satisfied: colorama in c:\users\beomj\anaconda3\lib\site-packages (from crayons->webdriver-manager) (0.4.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (1.25.9)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\beomj\anaconda3\lib\site-packages (from requests->webdriver-manager) (2020.6.20)
    


```python
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import exceptions
```


```python
addr = "https://www.imdb.com/find?q=deadpool&ref_=nv_sr_sm" 

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(addr)

reviews = []
rates = []
```

    [WDM] - Current google-chrome version is 84.0.4147
    [WDM] - Get LATEST driver version for 84.0.4147
    [WDM] - Driver [C:\Users\beomj\.wdm\drivers\chromedriver\win32\84.0.4147.30\chromedriver.exe] found in cache
     
    


    ---------------------------------------------------------------------------

    SessionNotCreatedException                Traceback (most recent call last)

    <ipython-input-3-81a120a4deea> in <module>
          1 addr = "https://www.imdb.com/find?q=deadpool&ref_=nv_sr_sm"
          2 
    ----> 3 driver = webdriver.Chrome(ChromeDriverManager().install())
          4 driver.get(addr)
          5 
    

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
df = pd.read_excel("영화raw데이터.xlsx")
names = df.movieNmEn
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-c6a18bd06d3d> in <module>
    ----> 1 df = pd.read_excel("영화raw데이터.xlsx")
          2 names = df.movieNmEn
    

    NameError: name 'pd' is not defined



```python
def find(name):
    try:
        driver.find_element_by_xpath("/html/body/div[2]/nav/div[2]/div[1]/form/div[2]/div/input").send_keys(name)
        driver.find_element_by_id("suggestion-search-button").click()      
        time.sleep(1)
        #------------------
        temp = driver.find_element_by_class_name("findList")
        temp = temp.find_elements_by_class_name('result_text')[0]   
        temp = temp.find_element_by_tag_name("a")
        temp.click()
        time.sleep(1)
        try:
            rates.append(driver.find_element_by_class_name("imdbRating").text)
        except:
            rates.append("")
        try:
            #------------------
            driver.find_element_by_link_text("USER REVIEWS").click()
            time.sleep(1)
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
        driver.get(addr)


```


```python
cnt = 0
for name in names:
    find(name)
    cnt +=1
    print(cnt, reviews[-1][:10], rates[-1])

```

    er Ho 4.9/10
    2,812
    4589 I found th 4.7/10
    2,911
    4590 Daniel Cra 8.0/10
    569,178
    4591 Well yes,  6.9/10
    236,836
    4592  
    4593  
    4594 The rock s 4.3/10
    565
    4595 I always w 4.1/10
    3,559
    4596 A word to  6.1/10
    7,402
    4597 The movie  5.7/10
    1,260
    4598 Shannon Tw 4.3/10
    512
    4599  
    4600  
    4601 This movie 4.3/10
    3,254
    4602 Though it  5.9/10
    14,946
    4603 One of the 8.5/10
    1,157,550
    4604 There's a  7.2/10
    23,649
    4605 6 years af 4.4/10
    8,959
    4606  
    4607 In The Fir 6.8/10
    116,207
    4608 In watchin 5.8/10
    16,897
    4609 Today I ha 4.5/10
    42,524
    4610 It would h 6.8/10
    142,754
    4611 ////////// 7.4/10
    266,320
    4612 Lots of bl 6.3/10
    67,198
    4613 If you did 7.2/10
    6,995
    4614 So I final 6.4/10
    12,023
    4615 ///There s 5.6/10
    31,686
    4616 ///I have  5.0/10
    8,977
    4617 I see many 6.3/10
    167,280
    4618 This was t 5.2/10
    94,953
    4619 Romero is  7.9/10
    113,046
    4620  
    4621 If you ask 7.1/10
    5,354
    4622 I saw this 6.4/10
    200,444
    4623 I honestly 5.9/10
    76,634
    4624 First off, 6.9/10
    62,354
    4625 First off, 5.5/10
    93,701
    4626 80/100. Th 7.4/10
    88,126
    4627 My friend  4.8/10
    13,913
    4628 I quite li 6.8/10
    68,844
    4629 Even thoug 6.6/10
    148,720
    4630 I'll keep  6.5/10
    83,972
    4631 Well, if I 6.7/10
    288,972
    4632 May be Bri 8.2/10
    11,082
    4633 Finally, a 6.9/10
    187,029
    4634 I really d 5.8/10
    36,342
    4635 Ice age 3  6.9/10
    218,382
    4636 Sara Campb 5.9/10
    58,422
    4637 It was a p 5.4/10
    7,259
    4638 In the wak 6.3/10
    17,195
    4639 ////////// 7.6/10
    457,093
    4640 It is a pi 5.2/10
    32,901
    4641  
    4642 This movie 5.6/10
    127,779
    4643 Once every 6.7/10
    225,254
    4644 This secon 6.0/10
    378,898
    4645 "Man About 5.5/10
    7,534
    4646 This is a  6.2/10
    183,216
    4647 How refres 5.8/10
    72,369
    4648 I FELT thi 7.1/10
    3,109
    4649 ///And by  6.5/10
    184,936
    4650 I saw a tr 6.7/10
    35,269
    4651 Hyped to t 5.8/10
    104,172
    4652 ///When I  6.0/10
    172,850
    4653 To be hone 6.3/10
    6,548
    4654 Terminator 6.5/10
    338,760
    4655 "U Turn" s 6.8/10
    48,601
    4656 I am sorry 6.7/10
    267,537
    4657 Scary Movi 5.1/10
    111,915
    4658 ///I was 2 7.8/10
    222,014
    4659 Wesley Sni 4.8/10
    4,090
    4660 There are  6.6/10
    459,200
    4661 Be warned, 5.8/10
    384
    4662 I saw this 7.5/10
    96,950
    4663 /////////O 6.2/10
    217,438
    4664 I disagree 6.7/10
    352
    4665 Despite it 7.1/10
    51,219
    4666  
    4667  
    4668  
    4669 ///Martin  5.5/10
    98,753
    4670 Dr. Megan  7.1/10
    14,767
    4671 You'd thin 6.1/10
    47,264
    4672 I watch a  5.9/10
    73,505
    4673 The tone o 4.9/10
    7,653
    4674 If you lis 6.5/10
    72,210
    4675 Suspension 6.8/10
    137,280
    4676 Wesley Sni 5.0/10
    4,688
    4677 Everybody  6.7/10
    21,683
    4678 Once every 7.3/10
    104,055
    4679 Friday the 6.5/10
    115,309
    4680 ///Uma Thu 5.6/10
    22,920
    4681 Let me jus 2.5/10
    69,174
    4682 ///I'm an  5.6/10
    21,394
    4683 As the las 7.1/10
    65,589
    4684 The Wrestl 7.9/10
    286,025
    4685 Watchmen a 8.1/10
    75,955
    4686 In every a 6.7/10
    46,770
    4687 ///Having  4.8/10
    48,925
    4688 I was pers 6.5/10
    90,045
    4689 ///Revolut 7.3/10
    191,538
    4690 After read 7.1/10
    143,980
    4691 As a fan o 5.3/10
    51,652
    4692 //////OK,  4.8/10
    56,676
    4693 It is hard 5.6/10
    776
    4694 /////////T 7.5/10
    119,656
    4695  
    4696 I gave thi 4.1/10
    384
    4697 This is th 7.4/10
    186,900
    4698 Monster Ho 6.6/10
    97,400
    4699 Quite an i 3.9/10
    546
    4700  
    4701 /////////W 5.7/10
    5,321
    4702 Still wond 6.1/10
    35,345
    4703 Wow! I'm s 5.5/10
    4,049
    4704 What are p 5.3/10
    40,241
    4705 I am a ter 7.6/10
    282,022
    4706 Once again 7.3/10
    16,652
    4707 A full six 6.7/10
    22,183
    4708 Mo (Brenda 6.1/10
    72,299
    4709 It does no 5.7/10
    10,887
    4710 This is on 6.0/10
    8,254
    4711 Well, I re 6.2/10
    1,332
    4712 The film i 5.6/10
    264
    4713 "Defiance" 7.2/10
    135,874
    4714 It's hard  6.6/10
    199,650
    4715 I just ret 6.6/10
    24,539
    4716 Fans of Sc 5.2/10
    25,434
    4717 This is on 5.9/10
    9,687
    4718 I have to  6.1/10
    145,682
    4719 I don't ca 6.2/10
    20,016
    4720 ///But WHY 7.1/10
    357,236
    4721 Red Dragon 7.2/10
    245,950
    4722 Infected w 4.4/10
    78,706
    4723 ///Since I 5.8/10
    64,665
    4724 This was g 5.8/10
    34,437
    4725 Although I 6.5/10
    21,344
    4726 Ulu Grosba 6.4/10
    12,719
    4727  
    4728  
    4729 Bulworth w 6.8/10
    24,408
    4730 I thought  6.4/10
    6,693
    4731 I suspect  7.3/10
    10,130
    4732 I have see 6.4/10
    24,091
    4733 It's hard  6.4/10
    195,437
    4734 I have see 5.0/10
    69,997
    4735 The image  6.9/10
    15,830
    4736 ///The sur 8.1/10
    893,064
    4737 Tony Scott 7.7/10
    323,381
    4738  
    4739 The proble 5.9/10
    122,564
    4740 I don't th 6.5/10
    238,324
    4741 I hate wre 6.3/10
    69,540
    4742 After the  8.5/10
    174,375
    4743 This movie 6.6/10
    178,539
    4744 The large  7.0/10
    22,078
    4745 My boyfrie 7.0/10
    324,016
    4746 Well...I r 4.9/10
    955
    4747 When I sta 7.6/10
    6,952
    4748 This movie 7.0/10
    6,664
    4749 Watching t 5.2/10
    74,203
    4750 Bourne is  7.7/10
    426,697
    4751 I am not s 6.7/10
    211,916
    4752 This is th 5.4/10
    19,763
    4753 I feel com 6.8/10
    92,113
    4754 Let's be c 6.7/10
    186,674
    4755 I must say 6.2/10
    279
    4756 Scanners I 5.3/10
    2,695
    4757 ///That mo 7.3/10
    674,375
    4758  
    4759 Pacific He 6.4/10
    17,503
    4760 Robert De  6.7/10
    143,775
    4761  
    4762 Critics an 5.1/10
    42,359
    4763  
    4764 ///Runaway 5.6/10
    91,103
    4765 This is on 5.7/10
    565
    4766 I don't kn 7.3/10
    643,566
    4767 Ahh. White 6.7/10
    19,180
    4768 You may lo 5.9/10
    36,908
    4769 One of the 7.2/10
    409,458
    4770 "Half Ligh 6.0/10
    13,120
    4771 Kimberly C 6.2/10
    149,087
    4772 Certainly, 6.2/10
    97,820
    4773 Ron Howard 6.7/10
    69,944
    4774 ///COME ON 4.7/10
    43,808
    4775 This is a  5.9/10
    9,708
    4776 I was tota 6.0/10
    268,055
    4777 How desper 4.5/10
    407
    4778 "We have t 6.4/10
    31
    4779 It's a mat 8.0/10
    667,103
    4780 I saw this 8.0/10
    45
    4781  
    4782 I've never 7.7/10
    296,555
    4783 So the dir 4.6/10
    610
    4784 If you lik 6.1/10
    2,821
    4785 I'll admit 5.1/10
    15,246
    4786 And thank  7.0/10
    7,424
    4787  6.8/10
    185
    4788  
    4789 "In the Cu 5.3/10
    21,520
    4790  6.4/10
    1,108
    4791  
    4792 I remember 5.3/10
    2,460
    4793 I noticed  5.6/10
    4,348
    4794 Raw Americ 7.3/10
    5,916
    4795 F/X2 was a 5.9/10
    8,623
    4796 Usually, r 6.2/10
    22,091
    4797 It has a f 6.3/10
    78,869
    4798 With nothi 6.3/10
    24,547
    4799 Didn't exp 3.9/10
    110
    4800 Sometimes  6.5/10
    4,486
    4801 I remember 6.2/10
    62,602
    4802 The story' 6.0/10
    124,352
    4803 I've been  6.2/10
    122,337
    4804 Wow, there 4.9/10
    150,838
    4805 ///Some mo 8.0/10
    353,734
    4806 This was o 5.8/10
    51,250
    4807 Certainly  7.0/10
    383,932
    4808 Some peopl 6.6/10
    213,268
    4809 Please do  5.5/10
    188
    4810 My cable p 3.9/10
    591
    4811  
    4812 I really g 6.2/10
    9,116
    4813 Who would  6.2/10
    90,073
    4814 I have to  6.2/10
    32,441
    4815 I get such 5.9/10
    10,260
    4816 This film, 6.5/10
    3,967
    4817 I could no 5.8/10
    8,654
    4818 "Town and  4.5/10
    4,622
    4819 I can see  6.1/10
    587
    4820 After hear 4.1/10
    15,033
    4821 After the  7.2/10
    59,401
    4822 This is a  7.5/10
    96,580
    4823 This is a  6.1/10
    49,678
    4824 This was a 7.4/10
    96,118
    4825 It's a sha 6.1/10
    40,984
    4826 It's hard  8.1/10
    505,432
    4827 The Omen i 7.5/10
    104,650
    4828 Unbeknowis 6.7/10
    156,299
    4829 ///To the  5.8/10
    4,526
    4830 Even thoug 6.4/10
    264
    4831 What's goi 4.7/10
    1,409
    4832 I want thi 6.6/10
    6,105
    4833 Some time  4.4/10
    1,042
    4834 Good as th 6.4/10
    236,106
    4835 This movie 4.7/10
    10,648
    4836 I was chom 6.5/10
    51,943
    4837 It all sta 5.7/10
    96,618
    4838 I am a big 4.5/10
    411
    4839 Abel Ferra 7.0/10
    30,936
    4840 In Manhatt 5.4/10
    52,896
    4841 Every acti 7.3/10
    123,168
    4842  
    4843  
    4844 I know I'm 7.7/10
    380,808
    4845 I've read  6.6/10
    392,332
    4846 Fans of `r 6.5/10
    6,289
    4847 ///I'm not 5.8/10
    128,131
    4848 Saying thi 6.3/10
    179,664
    4849 This is a  6.1/10
    52,964
    4850 Chris Rock 5.5/10
    21,375
    4851 It takes a 6.2/10
    159,593
    4852 Personally 5.3/10
    132
    4853 The lives  6.5/10
    7,641
    4854 Is a rough 6.5/10
    108,607
    4855 In this fi 6.8/10
    176,084
    4856 Nick Chen  6.1/10
    17,234
    4857 I have see 8.1/10
    74,648
    4858 //////I am 7.9/10
    23,408
    4859 SYNOPSIS:  7.6/10
    10,486
    4860 I saw this 4.0/10
    933
    4861 'Jack the  6.5/10
    3,877
    4862 Morris Che 5.1/10
    5,613
    4863 The third  6.5/10
    95,111
    4864 With its s 6.5/10
    8,866
    4865 'American  5.7/10
    2,282
    4866  
    4867 As he did  6.8/10
    13,021
    4868 Jeremy Iro 6.8/10
    8,486
    4869 I just lov 5.6/10
    47,874
    4870  
    4871 I'm guessi 6.9/10
    271,944
    4872 ///What is 6.6/10
    720
    4873 This is an 6.2/10
    15,337
    4874 Firstly, I 4.0/10
    899
    4875 Holy smoke 4.2/10
    721
    4876 This is ea 3.8/10
    1,342
    4877 This was a 4.1/10
    504
    4878 At the hei 4.6/10
    21,813
    4879 Undeniably 6.8/10
    73,458
    4880 ///This is 8.8/10
    184,312
    4881 I'm a big  5.1/10
    2,474
    4882 I've alway 2.7/10
    571
    4883 It is rare 6.6/10
    28,088
    4884 After seei 6.9/10
    19,859
    4885  
    4886  
    4887 "Sliver" w 5.1/10
    28,488
    4888 Jean Renoi 7.6/10
    2,830
    4889 `Woman on  6.4/10
    1,880
    4890 MALICE is  6.4/10
    22,138
    4891 `Half Man, 6.9/10
    9,290
    4892 This movie 7.0/10
    15,887
    4893 I am in Wa 7.6/10
    10,105
    4894 This is a  6.9/10
    9,367
    4895 Mrs. Marti 5.6/10
    1,734
    4896 Easily not 6.9/10
    323,833
    4897 It's no re 6.3/10
    23,123
    4898  
    4899  
    4900 Note: I st 7.6/10
    331,081
    4901 I just wan 6.3/10
    1,082
    4902 I really e 5.7/10
    80,171
    4903 I can't be 6.5/10
    62,372
    4904 For the mo 6.7/10
    117,003
    4905 To the con 5.3/10
    662
    4906 Well, I ju 6.8/10
    11,929
    4907 I thought  5.6/10
    72,677
    4908 I love thi 6.4/10
    50,652
    4909 Any compar 5.4/10
    42,798
    4910 The acting 7.3/10
    59,823
    4911 ///Hidalgo 6.7/10
    76,313
    4912 I don't re 5.3/10
    14,599
    4913 Although I 5.2/10
    16,775
    4914 This movie 5.6/10
    26,346
    4915 I didn't e 6.5/10
    76,158
    4916 The lovabl 4.9/10
    40,365
    4917 While stay 6.5/10
    55,944
    4918 I often fi 8.0/10
    407,836
    4919 The Lizzie 5.5/10
    35,108
    4920 First revi 8.2/10
    20,536
    4921 Slow in a  8.2/10
    19,201
    4922 I went int 5.7/10
    50,540
    4923 "Syriana"  6.9/10
    124,336
    4924 In preview 6.2/10
    132,392
    4925 I watch th 8.2/10
    1,008,026
    4926 I read her 6.5/10
    51,320
    4927 Before you 6.1/10
    17,322
    4928 This was a 5.7/10
    84,053
    4929 As a teena 5.5/10
    53,568
    4930 Deep Throa 6.7/10
    6,434
    4931  
    4932 A dream se 6.4/10
    5,918
    4933 Old wounds 6.4/10
    58,644
    4934 Highly und 6.4/10
    133,818
    4935 Isabel (Gr 4.8/10
    391
    4936 I applaud  6.1/10
    12,410
    4937  4.0/10
    12
    4938 Be honest: 7.3/10
    44,310
    4939 As a young 5.1/10
    30,027
    4940 i guess i  5.5/10
    27,877
    4941 I believe  8.6/10
    23,804
    4942 Back in th 3.3/10
    3,995
    4943 It seems t 7.3/10
    379,624
    4944 This movie 6.0/10
    85,485
    4945 This is no 5.3/10
    1,084
    4946 ///The fil 7.6/10
    471,417
    4947 I guess it 6.6/10
    82,822
    4948 I like sla 3.7/10
    540
    4949 Where Slee 5.8/10
    42
    4950 We lived t 7.7/10
    279,024
    4951  
    4952 I rented K 7.0/10
    25,786
    4953 Possibly J 6.6/10
    20,999
    4954 Surprising 5.5/10
    3,127
    4955 Outside Pr 6.4/10
    8,425
    4956 A stunning 7.7/10
    83,287
    4957 Having fai 5.3/10
    25,479
    4958 I must con 6.8/10
    11,734
    4959 This movie 7.3/10
    12,193
    4960 BLESS THE  5.1/10
    13,729
    4961 Okay, this 3.1/10
    1,827
    4962 Confidentl 9.0/10
    2,236,662
    4963  
    4964  
    4965  
    4966  
    4967 Producer J 5.7/10
    105,112
    4968 ///Oh man! 6.6/10
    115,922
    4969 What I lik 6.6/10
    14,840
    4970 Captain Da 6.3/10
    43,657
    4971 If you can 5.0/10
    2,748
    4972 ///Full Sy 3.9/10
    559
    4973 In year 20 3.9/10
    4,318
    4974  
    4975 This film  6.3/10
    270
    4976  
    4977 It was nic 6.4/10
    41,615
    4978 Sebastian  5.8/10
    121,709
    4979 This TV se 4.2/10
    158
    4980 ///First o 2.6/10
    24,753
    4981 Usually, w 6.2/10
    38,920
    4982 I was fort 6.9/10
    8,818
    4983 Why do the 5.5/10
    161,420
    4984  
    4985  
    4986 Being luck 6.8/10
    329,385
    4987 We all kno 7.2/10
    82,242
    4988 But the ma 5.4/10
    120,856
    4989 I finally  4.3/10
    15,970
    4990 Great fun! 6.6/10
    117,448
    4991 I know a l 3.6/10
    21,179
    4992 After read 6.6/10
    58,013
    4993 Yes I know 6.4/10
    153,772
    4994 Even thoug 6.0/10
    11,957
    4995 At what po 5.4/10
    3,053
    4996 ///The Pat 7.2/10
    250,470
    4997 "The Crow: 4.9/10
    10,679
    4998 Airborne i 6.3/10
    6,227
    4999 William Ba 4.8/10
    325
    5000 This is an 5.7/10
    21,270
    5001  6.8/10
    5
    5002 I would ho 6.6/10
    35,959
    5003 ///My spou 2.5/10
    75,145
    5004 Brian De P 7.1/10
    379,719
    5005 There have 6.3/10
    49,288
    5006 The story  5.5/10
    11,345
    5007 Nineteen n 7.4/10
    91,742
    5008 Absolutely 3.9/10
    9,381
    5009 There is m 4.6/10
    11,105
    5010 The whole  5.3/10
    9,139
    5011 ///Though  5.6/10
    30,588
    5012 This is a  6.7/10
    8,903
    5013 While most 7.1/10
    222,363
    5014 Tony D'Ama 6.9/10
    110,563
    5015 I stumbled 6.3/10
    10,208
    5016 (10 out of 6.7/10
    84,242
    5017 As I start 6.6/10
    2,201
    5018 American G 7.8/10
    383,253
    5019 I don't kn 6.1/10
    1,556
    5020 Let's get  6.2/10
    42,001
    5021 I saw this 6.1/10
    2,646
    5022  
    5023 The legend 7.2/10
    673,476
    5024 ///Reminis 6.8/10
    68,134
    5025  5.9/10
    166,595
    5026  
    5027 I caught t 6.6/10
    2,856
    5028 I found th 6.3/10
    158,693
    5029 Saw this S 6.5/10
    12,890
    5030 I have utm 7.6/10
    371,459
    5031 It takes a 6.2/10
    159,593
    5032 Charlie Ka 7.7/10
    304,173
    5033 A movie fi 3.6/10
    2,431
    5034 This movie 6.3/10
    16,828
    5035 For Love O 6.6/10
    31,232
    5036 While I wa 6.4/10
    12,061
    5037 problem: w 5.1/10
    253
    5038 This movie 7.3/10
    232
    5039 ///Reading 6.1/10
    16,120
    5040 Scream 3 i 5.6/10
    120,968
    5041 One of the 7.1/10
    20,995
    5042 The storyl 5.8/10
    736
    5043 Part of th 7.7/10
    35,509
    5044 The Boondo 7.8/10
    224,550
    5045 I feel a n 6.0/10
    2,476
    5046 The first  4.9/10
    11,682
    5047 This movie 3.8/10
    101
    5048 This film  6.6/10
    7,696
    5049 I saw this 6.0/10
    12,709
    5050 I think th 5.4/10
    6,514
    5051 Let me beg 5.8/10
    19,706
    5052 Year of th 5.6/10
    2,349
    5053 This is a  4.6/10
    476
    5054 "Chasing A 7.2/10
    130,640
    5055 Jennifer R 3.2/10
    585
    5056 Black Dog  5.5/10
    8,285
    5057 ///In the  7.0/10
    91,081
    5058 This is my 6.2/10
    18,114
    5059 There is m 4.6/10
    11,105
    5060 Okay sure  5.7/10
    52,492
    5061 This movie 7.1/10
    49,089
    5062 Haunted by 6.8/10
    97,293
    5063 The filmma 4.6/10
    674
    5064 All true H 4.3/10
    42,855
    5065 I really e 6.7/10
    35,624
    5066 What I lik 5.4/10
    20,493
    5067 Charlie Sh 4.9/10
    1,389
    5068 i thought  6.2/10
    63,466
    5069 Just watch 6.1/10
    2,741
    5070 My two fav 6.9/10
    43,556
    5071 Please, ne 4.3/10
    439
    5072 This is on 6.4/10
    7,519
    5073 I got drag 7.9/10
    3,616
    5074 When this  5.5/10
    12,151
    5075 Stir of Ec 7.0/10
    73,287
    5076 I taped th 6.9/10
    25,765
    5077 While the  6.5/10
    17,443
    5078  
    5079 This is a  6.5/10
    82
    5080 Seventies  5.2/10
    581
    5081 I don't ho 5.5/10
    22,220
    5082 **SPOILERS 5.4/10
    1,564
    5083 This lovel 7.1/10
    73,302
    5084 Q&A is one 7.2/10
    88
    5085 I wouldn't 4.6/10
    6,396
    5086 I wasn't g 7.6/10
    90,023
    5087 Any film t 6.9/10
    7,362
    5088 In 1976, i 6.8/10
    14,842
    5089 //////'Hom 7.6/10
    450,846
    5090 It's just  5.2/10
    40,109
    5091 i expected 8.6/10
    1,110,076
    5092 ///Clay Pi 6.6/10
    10,533
    5093 Sandwiched 6.6/10
    397,110
    5094 ///After v 8.0/10
    44,595
    5095  
    5096 So many ne 6.2/10
    46,793
    5097 I had the  6.4/10
    17,730
    5098 ///I went  7.0/10
    115,957
    5099 ///This mo 6.7/10
    48,940
    5100 Wow. Shoot 6.7/10
    143,033
    5101 One of the 6.9/10
    230,930
    5102 Amy (Kate  6.2/10
    92,925
    5103 I saw this 6.2/10
    23,376
    5104 //////We w 5.9/10
    73,713
    5105 I just got 7.4/10
    92,192
    5106  6.0/10
    12
    5107 Directed a 6.0/10
    2,312
    5108 To compare 5.8/10
    112,856
    5109 Not expect 6.8/10
    216,846
    5110 Very Enter 6.5/10
    40,778
    5111  
    5112 "No Reserv 6.3/10
    69,459
    5113 I knew wha 7.5/10
    119,483
    5114 Return To  6.9/10
    13,594
    5115 //////Let  7.3/10
    304,697
    5116 We went to 4.8/10
    27,804
    5117 I thought  6.5/10
    11,670
    5118 I went to  7.6/10
    249,824
    5119 Deep Risin 6.1/10
    32,687
    5120 If you're  5.8/10
    7,984
    5121 While doin 6.8/10
    251,760
    5122 Cronenberg 7.4/10
    220,169
    5123 I don't un 5.4/10
    137,308
    5124 I give thi 8.5/10
    1,012,633
    5125 When I fir 7.5/10
    493,079
    5126 I did not  6.8/10
    50,008
    5127 This is no 6.5/10
    7,685
    5128 Wild Thing 6.5/10
    105,956
    5129 ///If one  6.3/10
    50,221
    5130 The movie  5.7/10
    1,648
    5131 ... his ve 5.5/10
    2,664
    5132 ///Sequel  4.4/10
    5,608
    5133 I'm wonder 5.4/10
    24,580
    5134  
    5135  
    5136  
    5137  
    5138  
    5139  
    5140  
    5141  
    5142  
    5143  
    5144  
    5145  
    5146  
    5147  
    5148  
    5149  
    5150  
    5151  
    5152  
    5153  
    5154  
    5155  
    5156  
    5157  
    5158  
    5159  
    5160  
    5161  
    5162  
    5163  
    5164  
    5165  
    5166  
    5167  
    5168  
    5169  
    5170  
    5171  
    5172  
    5173  
    5174  
    5175  
    5176  
    5177  
    5178  
    5179  
    5180  
    5181  
    5182  
    5183  
    5184  
    5185  
    5186  
    5187  
    5188  
    5189  
    5190  
    5191  
    5192  
    5193  
    5194  
    5195  
    5196  
    5197  
    5198  
    5199  
    5200  
    5201  
    5202  
    5203  
    5204  
    5205  
    5206  
    5207  
    5208  
    5209  
    5210  
    5211  
    5212  
    5213  
    5214  
    5215  
    5216  
    5217  
    5218  
    5219  
    5220  
    5221  
    5222  
    5223  
    5224  
    5225  
    5226  
    5227  
    5228  
    5229  
    5230  
    5231  
    5232  
    5233  
    5234  
    5235  
    5236  
    5237  
    5238  
    5239  
    5240  
    5241  
    5242  
    5243  
    5244  
    5245  
    5246  
    5247  
    5248  
    5249  
    5250  
    5251  
    5252  
    5253  
    5254  
    5255  
    5256  
    5257  
    5258  
    5259  
    5260  
    5261  
    5262  
    5263  
    5264  
    5265  
    5266  
    5267  
    5268  
    5269  
    5270  
    5271  
    5272  
    5273  
    5274  
    5275  
    5276  
    5277  
    5278  
    5279  
    5280  
    5281  
    5282  
    5283  
    5284  
    5285  
    5286  
    5287  
    5288  
    5289  
    5290  
    5291  
    5292  
    5293  
    5294  
    5295  
    5296  
    5297  
    5298  
    5299  
    5300  
    5301  
    5302  
    5303  
    5304  
    5305  
    5306  
    5307  
    5308  
    5309  
    5310  
    5311  
    5312  
    5313  
    5314  
    5315  
    5316  
    5317  
    5318  
    5319  
    5320  
    5321  
    5322  
    5323  
    5324  
    5325  
    5326  
    5327  
    5328  
    5329  
    5330  
    5331  
    5332  
    5333  
    5334  
    5335  
    5336  
    5337  
    5338  
    5339  
    5340  
    5341  
    5342  
    5343  
    5344  
    5345  
    5346  
    5347  
    5348  
    5349  
    5350  
    5351  
    5352  
    5353  
    5354  
    5355  
    5356  
    5357  
    5358  
    5359  
    5360  
    5361  
    5362  
    5363  
    5364  
    5365  
    5366  
    5367  
    5368  
    5369  
    5370  
    5371  
    5372  
    5373  
    5374  
    5375  
    5376  
    5377  
    5378  
    5379  
    5380  
    5381  
    5382  
    5383  
    5384  
    5385  
    5386  
    5387  
    5388  
    5389  
    5390  
    5391  
    5392  
    5393  
    5394  
    5395  
    5396  
    5397  
    5398  
    5399  
    5400  
    5401  
    5402  
    5403  
    5404  
    5405  
    5406  
    5407  
    5408  
    5409  
    5410  
    5411  
    5412  
    5413  
    5414  
    5415  
    5416  
    5417  
    5418  
    5419  
    5420  
    5421  
    5422  
    5423  
    5424  
    5425  
    5426  
    5427  
    5428  
    5429  
    5430  
    5431  
    5432  
    5433  
    5434  
    5435  
    5436  
    5437  
    5438  
    5439  
    5440
    5441  
    5442  
    5443  
    5444  
    5445  
    5446  
    5447  
    5448  
    5449  
    5450  
    5451  
    5452  
    5453  
    5454  
    5455  
    5456  
    5457  
    5458  
    5459  
    5460  
    5461  
    5462  
    5463  
    5464  
    5465  
    5466  
    5467  
    5468  
    5469  
    5470  
    5471  
    5472  
    5473  
    5474  
    5475  
    5476  
    5477  
    5478  
    5479  
    5480  
    5481  
    5482  
    5483  
    5484  
    5485  
    5486  
    5487  
    5488  
    5489  
    5490  
    5491  
    5492  
    5493  
    5494  
    5495  
    5496  
    5497  
    5498  
    5499  
    5500  
    5501  
    5502  
    5503  
    5504  
    5505  
    5506  
    5507  
    5508  
    5509  
    5510  
    5511  
    5512  
    5513  
    5514  
    5515  
    5516  
    5517  
    5518  
    5519  
    5520  
    5521  
    5522  
    5523  
    5524  
    5525  
    5526  
    5527  
    5528  
    5529  
    5530  
    5531  
    5532  
    5533  
    5534  
    5535  
    5536  
    5537  
    5538  
    5539  
    5540  
    5541  
    5542  
    5543  
    5544  
    5545  
    5546  
    5547  
    5548  
    5549  
    5550  
    5551  
    5552  
    5553  
    5554  
    5555  
    5556  
    5557  
    5558  
    5559  
    5560  
    5561  
    5562  
    5563  
    5564  
    5565  
    5566  
    5567  
    5568  
    5569  
    5570  
    5571  
    5572  
    5573  
    5574  
    5575  
    5576  
    5577  
    5578  
    5579  
    5580  
    5581  
    5582  
    5583  
    5584  
    5585  
    5586  
    5587  
    5588  
    5589  
    5590  
    5591  
    5592  
    5593  
    5594  
    5595  
    5596  
    5597  
    5598  
    5599  
    5600  
    5601  
    5602  
    5603  
    5604  
    5605  
    5606  
    5607  
    5608  
    5609  
    5610  
    5611  
    5612  
    5613  
    5614  
    5615  
    5616  
    5617  
    5618  
    5619  
    5620  
    5621  
    5622  
    5623  
    5624  
    5625  
    5626  
    5627  
    5628  
    5629  
    5630  
    5631  
    5632  
    5633  
    5634  
    5635  
    5636  
    5637  
    5638  
    5639  
    5640  
    5641  
    5642  
    5643  
    5644  
    5645  
    5646  
    5647  
    5648  
    5649  
    5650  
    5651  
    5652  
    5653  
    5654  
    5655  
    5656  
    5657  
    5658  
    5659  
    5660  
    5661  
    5662  
    5663  
    5664  
    5665  
    5666  
    5667  
    5668  
    5669  
    5670  
    5671  
    5672  
    5673  
    5674  
    5675  
    5676  
    5677  
    5678  
    5679  
    5680  
    5681  
    5682  
    5683  
    5684  
    5685  
    5686  
    5687  
    5688  
    5689  
    5690  
    5691  
    5692  
    5693  
    5694  
    5695  
    5696  
    5697  
    5698  
    5699  
    5700  
    5701  
    5702  
    5703  
    5704  
    5705  
    5706  
    5707  
    5708  
    5709  
    5710  
    5711  
    5712  
    5713  
    5714  
    5715  
    5716  
    5717  
    5718  
    5719  
    5720  
    5721  
    5722  
    5723  
    5724  
    5725  
    5726  
    5727  
    5728  
    5729  
    5730  
    5731  
    5732  
    5733  
    5734  
    5735  
    5736  
    5737  
    5738  
    5739  
    5740  
    5741  
    5742  
    5743  
    5744  
    5745  
    5746  
    5747  
    5748  
    5749  
    5750  
    5751  
    5752  
    5753  
    5754  
    5755  
    5756  
    5757  
    5758  
    5759  
    5760  
    5761  
    5762  
    5763  
    5764  
    5765  
    5766  
    5767  
    5768  
    5769  
    5770  
    5771  
    5772  
    5773  
    5774  
    5775  
    5776  
    5777  
    5778  
    5779  
    5780  
    5781  
    5782  
    5783  
    5784  
    5785  
    5786  
    5787  
    5788  
    5789  
    5790  
    5791  
    5792  
    5793  
    5794  
    5795  
    5796  
    5797  
    5798  
    5799  
    5800  
    5801  
    5802  
    5803  
    5804  
    5805  
    5806  
    5807  
    5808  
    5809  
    5810  
    5811  
    5812  
    5813  
    5814  
    5815  
    5816  
    5817  
    5818  
    5819  
    5820  
    5821  
    5822  
    5823  
    5824  
    5825  
    


```python
data = pd.DataFrame({'영화명EN':names[:len(reviews)], '리뷰':reviews, '평점':rates})
data.to_excel("1970_2019_review_en.xlsx", index=False)
```


```python
data[:10]
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
      <th>리뷰</th>
      <th>평점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Think Like a Dog</td>
      <td>Decent kids movie,predictable,somewhat forced ...</td>
      <td>5.1/10\n646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Mutants</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drive</td>
      <td>///One reviewer here suggested that instead of...</td>
      <td>7.8/10\n557,699</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jurassic Thunder</td>
      <td>....but then again, did you expect an award wi...</td>
      <td>1.9/10\n140</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sea Monsters2</td>
      <td>Loose plot with no content. The story is vauge...</td>
      <td>3.7/10\n30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Unhinged</td>
      <td>Ok, so I'm a fan of Russell Crowe. He has done...</td>
      <td>6.2/10\n2,728</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mulan</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>JOJO RABBIT</td>
      <td>This film was exceptional and one of the best ...</td>
      <td>7.9/10\n248,630</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Killerman</td>
      <td>Strong performances, intimidating villains, cl...</td>
      <td>5.4/10\n3,142</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tenet</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>영화명EN</th>
      <th>리뷰</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Think Like a Dog</td>
      <td>Decent kids movie,predictable,somewhat forced ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The New Mutants</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drive</td>
      <td>///One reviewer here suggested that instead of...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jurassic Thunder</td>
      <td>....but then again, did you expect an award wi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sea Monsters2</td>
      <td>Loose plot with no content. The story is vauge...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Unhinged</td>
      <td>Ok, so I'm a fan of Russell Crowe. He has done...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mulan</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>JOJO RABBIT</td>
      <td>This film was exceptional and one of the best ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Killerman</td>
      <td>Strong performances, intimidating villains, cl...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tenet</td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>John Wick Chapter Two</td>
      <td>In this 2nd installment of John Wick, the styl...</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
