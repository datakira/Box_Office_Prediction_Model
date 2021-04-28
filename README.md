![캡처1](https://user-images.githubusercontent.com/65406000/116097271-f4845680-a6e4-11eb-8c9e-3cf25c7a6b59.JPG)

# Box Office record Prediction Model
> 한국데이터산업진흥원 빅데이터 청년인재 프로젝트

> Final project at DataCampus - Yonsei University in 2020

> Team Name:   슬기로운 분석생활

> Team Member: 박범진 김윤하 백인진 이희지 정경희

국내 영화 산업이 성장함에 따라 외국영화를 수입하여 상영하려는 시도는 활발히 늘어나고 있다.

이 프로젝트에서는 **영화의 기본 정보**와 **리뷰 데이터**로
*수입된 영화의 흥행 정도를 예측하는 모델*을 구축한다.

![캡처3](https://user-images.githubusercontent.com/65406000/116101456-c99c0180-a6e8-11eb-8c6b-a813b2acfc27.JPG)







## Analytic Process

- 데이터 선정 및 수집


- 데이터 전처리


- 모델링





## 1. 데이터 선정 및 수집

지난 5년간 국내에서 제작된 영화를 제외하고 대부분의 관객수 점유율을 갖는 국가는 미국이다.

제작국가가 미국인 영화 중 한국에서 1970년~2020년 개봉한 영화의 데이터를 수집하였다.

> 수집 사이트: KOBIS, Mojo, IMDb, NAVER

> 수집 방법: 웹크롤링 & API


![캡처2](https://user-images.githubusercontent.com/65406000/116097769-678dcd00-a6e5-11eb-9ded-eddebf68bbd1.JPG)




## 2. 리뷰 수치화 및 시각화

### 1) 데이터 수치화
- 배우, 감독, 영화사를 수치화

배우: 영화의 제작년도와 같은 년도에 매겨진 배우점수들의 평균

감독: 감독의 최근 3년간의 영화 수익

영화사: 영화사의 시장 점유율

![캡처4](https://user-images.githubusercontent.com/65406000/116100394-d53af880-a6e7-11eb-8fd2-fb37821892a6.JPG)


### 2) 리뷰 수치화
- 리뷰에 언급된 '스토리, 음악, 배우'에 대한 빈도, 긍/부정을 수치화

![캡처5](https://user-images.githubusercontent.com/65406000/116100977-66aa6a80-a6e8-11eb-929f-3327192a8d1d.JPG)


### 3) 스토리 수치화
- 줄거리 PCA 및 클러스터링

![캡처8](https://user-images.githubusercontent.com/65406000/116259152-84db9d80-a7b0-11eb-9903-728e0febdf4f.JPG)


### 4) 결측치 처리
- Null 값에 대해 변수의 속성에 맞게 drop, 최빈값 대체, 0으로 대체, 예측값 대체 등 결측치 처리


### 5) 다중공선성 처리
**1. 영화 객관적 속성 데이터**

month, openDt_weather, openDt_quarter 은 각각 영화의 개봉월, 날씨, 분기이므로 연관성이 높음 -> **openDt_quarter** 선택

director_revenue와 Domestic 은 각각 감독의 수입과 한국에서의 영화 지출으로 연관성이 높음 -> **Domestic** 선택

**2. 영화 정성적 속성 데이터**

neg, pos, neu, compund 은 모든 토픽(배우, 연기, 연출, 노래, 스토리)에서 파생된 변수로 연관성이 높음 -> **compound** 선택


## 3. 모델링

### 1) 종속변수 등급화


영화의 관객수를 기준으로 A,B,C,D,F등급으로 등급화

A: 관객수 5백만 이상 (1481개)

B: 관객수 1백만 이상 (1134개)

C: 관객수 1십만 이상 (603개)

D: 관객수 1천명 이상 (239개)

F: 그 외 (33개)



### 2) 모델링 결과

리뷰 & 줄거리 데이터 포함 모델 vs 포함하지 않은 모델

![캡처7](https://user-images.githubusercontent.com/65406000/116102736-e97ff500-a6e9-11eb-9792-71ac5da957f4.JPG)






