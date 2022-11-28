# Ai-Project-Flower-Recognition-Model
## Topic
```
꽃 생장 파악 알고리즘(CNN, Classification etc)
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
- 영양분이 부족하거나, 물이 부족하거나 등등의 원인으로 인해 생기는 현상 팁번처럼, 지금 싱싱한지 아닌지를 파악 

2. 꽃 사진을 입력했을 때, 이게 꽃이 폈는지 아닌지 파악
(1) 잎 크기로 잎이 얼마나 성장했는지 구별
(2) 꽃 이미지로, 어떤 꽃인지 판별

## Dataset
```
https://docs.google.com/spreadsheets/d/1mdLbku2yM-XiBmN0Lm_O82xbFbpup1E1mkY1KXwGuds/edit#gid=0
https://www.kaggle.com/datasets/cf488efb70f71b0db8c5a69539ea35874787d4a4ab835126168e7af1723418d7
```
![images of strawberry](https://user-images.githubusercontent.com/117802301/204216971-6f71729a-33cc-4101-af74-6cf7dbff3470.png)

본 이미지는 kaggle 사이트에서 각 calciumdeficiency 사진 805장과 건강한 잎 626장을 데이터 셋으로 불러왔습니다. 
구글 드라이브를 통해 총 1433의 파일을 저장하여 이를 colab으로 불러와 꽃 사진을 분류하는 프로그램을 진행하였고 kaggle 사이트에서 가져온 1433의 사진 파일과 10 columns 뿐만 아니라
구글에서 따로 이미지 크롤링을 python 셀레니움을 통해 자동으로 정리하는 시스템을 갖보았습니다.

## Methodology
1) 파이썬 셀레니움을 통한 이미지 크롤링.
VSCODE를 사용하여 가상환경에서 셀레니움을 설치한다. 이후 구글에서 이미지 크롤링이 가능한 코드를 입력하여 검색어("Tipburn","Healthy leaf"),를 바꾸어가며 이미지를 수집하고, 사용 가능한 데이터를 정리한다.
## Evaluation & Analysis
### 1) 이미지 크롤링
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


### 2) Flower-Recognition-Model
#### (1) Model 1 
(1) parameter :
 1) image size : 150 X 150
 2) batch_size : 20
 3) model composition :

```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
 4) optimizers : RMSprop
 5) lr = 1e-4
 6) steps_per_epoch = 20 & epochs = 20 

![aix_model1](https://user-images.githubusercontent.com/117802301/204218717-473c62d5-7189-471a-923d-229cc7a63047.png)

```

```
```
```
```
```

## Conclusion : Discussion
```
```
## Related Works
https://goldsystem.tistory.com/822
https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic/report
https://www.kaggle.com/datasets/cf488efb70f71b0db8c5a69539ea35874787d4a4ab835126168e7af1723418d7
- 원영준 교수님 딥러닝 강의자료
