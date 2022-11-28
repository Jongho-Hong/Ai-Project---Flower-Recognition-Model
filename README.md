# Ai-Project-Flower-Recognition-Model
## Topic
```
## 꽃 생장 파악 알고리즘(CNN, Classification etc)
```
## Member
```
권석준, 기계공학부, jun115533@hanyang.ac.kr
김경휘, 기계공학부, kyunghui98@hanyang.ac.kr
홍종호, 기계공학부, hjho6389@hanyang.ac.kr
```
## Index
```
1. Motivation

2. Datasets

3. Methodology

4. Evaluation & Analysis

5. Conclusion : Dicussion

6. Related Works
```
## Motivation
세계적인 식량 문제로 인해, 스마트팜에 대한 관심이 늘어나고 있고 개인용 소형 스마트팜의 보급 및 여러 고부가가치 작물에 대한 기술의 수요가 증가하고 있는 상황입니다.
이러한 상황에서, 꽃의 생장 단계 및 생장 상황을 파악하는 딥러닝 알고리즘에 대한 개발을 시도해보고자 이번 프로젝트를 기획하게 되었습니다.

**스마트팜용 꽃 생장 파악 알고리즘**
1. 식물의 생장 상황을 파악 : 병들었는지 아닌지 구별
* 팁번 : 나뭇잎 끝이 누렇게 변하는 현상
- 영양분이 부족하거나, 물이 부족하거나 등등의 원인으로 인해 생기는 현상
팁번처럼, 지금 잘 자라고 있는지 아닌지(싱싱한지 아닌지)를 파악 

2. 꽃 사진을 입력했을 때, 이게 꽃이 폈는지 아닌지 파악
(1) 잎 크기로 잎이 얼마나 성장했는지 구별
(2) 꽃 이미지로, 어떤 꽃인지 판별

## Dataset
```
https://docs.google.com/spreadsheets/d/1mdLbku2yM-XiBmN0Lm_O82xbFbpup1E1mkY1KXwGuds/edit#gid=0
https://www.kaggle.com/datasets/cf488efb70f71b0db8c5a69539ea35874787d4a4ab835126168e7af1723418d7
```
## Methodology
1) 파이썬 셀레니움을 통한 이미지 크롤링.
VSCODE를 사용하여 가상환경에서 셀레니움을 설치한다. 이후 구글에서 이미지 크롤링이 가능한 코드를 입력하여 검색어("Tipburn","Healthy leaf"),를 바꾸어가며 이미지를 수집하고, 사용 가능한 데이터를 정리한다.
## Evaluation & Analysis
1) 이미지 크롤링
```
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from attr import Attribute
import urllib.request
driver = webdriver.Chrome()
driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
elem = driver.find_element(By.NAME, "q")
elem.send_keys("healthy leaf")
elem.send_keys(Keys.RETURN)
SCROLL_PAUSE_TIME = 1
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
        except:
            break
    last_height = new_height
images=driver.find_elements(By.CSS_SELECTOR,'.rg_i.Q4LuWd')
count=1
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl=driver.find_element(By.XPATH,"/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div/div[3]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
        urllib.request.urlretrieve(imgUrl,str(count)+".jpg")
        count=count+1
    except:
        pass
    
driver.close()
```
![20221126_205108](https://user-images.githubusercontent.com/117706557/204087357-12888ceb-1214-4917-9bcd-086558420832.png)

## Conclusion : Discussion
```
```
## Related Works
