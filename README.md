# Ai-Project-Flower-Recognition-Model
# Topic
```
꽃 생장 파악 알고리즘(CNN, Classification etc)
```
# Member
```
권석준, 기계공학부, jun115533@hanyang.ac.kr
김경휘, 기계공학부, kyunghui98@hanyang.ac.kr
홍종호, 기계공학부, hjho6389@hanyang.ac.kr
```
# Index
```
1. Motivation

2. Datasets

3. Methodology

4. Evaluation & Analysis

5. Result

6. Conclusion : Dicussion

7. Related Works
```
# Motivation
세계적인 식량 문제로 인해, 스마트팜에 대한 관심이 늘어나고 있고 개인용 소형 스마트팜의 보급 및 여러 고부가가치 작물에 대한 기술의 수요가 증가하고 있는 상황입니다.
이러한 상황에서, 꽃의 생장 단계 및 생장 상황을 파악하는 딥러닝 알고리즘에 대한 개발을 시도해보고자 이번 프로젝트를 기획하게 되었습니다.

**스마트팜용 꽃 생장 파악 알고리즘**
1. 식물의 생장 상황을 파악 : 병들었는지 아닌지 구별
* 팁번 : 나뭇잎 끝이 누렇게 변하는 현상
* 영양분이 부족하거나, 물이 부족하거나 등등의 원인으로 인해 생기는 현상 팁번처럼, 지금 싱싱한지 아닌지를 파악 

2. 꽃 사진을 입력했을 때, 이게 꽃이 폈는지 아닌지 파악
* 잎 크기로 잎이 얼마나 성장했는지 구별
* 꽃 이미지로, 어떤 꽃인지 판별

# Dataset
```
https://docs.google.com/spreadsheets/d/1mdLbku2yM-XiBmN0Lm_O82xbFbpup1E1mkY1KXwGuds/edit#gid=0
https://www.kaggle.com/datasets/cf488efb70f71b0db8c5a69539ea35874787d4a4ab835126168e7af1723418d7
```
![images of strawberry](https://user-images.githubusercontent.com/117802301/204216971-6f71729a-33cc-4101-af74-6cf7dbff3470.png)

본 이미지는 kaggle 사이트에서 각 calciumdeficiency 사진 805장과 건강한 잎 626장을 데이터 셋으로 불러왔습니다. 
구글 드라이브를 통해 총 1431의 파일을 저장하여 이를 colab으로 불러와 꽃 사진을 분류하는 프로그램을 진행하였고 kaggle 사이트에서 가져온 1431의 사진 파일과 10 columns 뿐만 아니라
구글에서 따로 이미지 크롤링을 python 셀레니움을 통해 자동으로 정리하는 시스템을 갖보았습니다.



# Methodology


## 1) Image crawling using Python Selenium

VSCODE를 사용하여 가상환경에서 셀레니움을 설치한다. 이후 구글에서 이미지 크롤링이 가능한 코드를 입력하여 검색어("Tipburn","Healthy leaf"),를 바꾸어가며 이미지를 수집하고, 사용 가능한 데이터를 정리한다.

### 실행모드
![imagecrawling](https://user-images.githubusercontent.com/117802301/204221170-55b3ad92-6994-43a8-9382-45d4b71bd4f4.gif)


## 2) Flow-Recognition Method using CNN

### (1) CNN이 무엇인가?
![CNN system](https://user-images.githubusercontent.com/117802301/204230213-1810f4cf-9469-4f28-b62e-81d46a8dc7af.gif)

CNN은 Convolutional Neural Networks의 약자로 합성곱 전처리 작업이 들어가는 신경망 모델입니다. 날 것의 이미지를 그대로 받으면서 공간적/지역적 정보를 유지한 채 특성들의 계층을 빌드업하는 형식으로 중요 포인트는 이미지 전체보다는 부분을 보는 것으로 이미지의 한 
픽셀과 주변 픽셀들의 연관성을 살립니다. 인간의 시신경 구조를 모방하는 기술인 CNN은 이미지를 인식하기 위해 패턴을 찾는데 특히 유용하며 데이터를 직접 학습하고 패턴을 사용해 이미지의 공간 정보를 유지한 채 학습을 하게 되는 모델입니다.

### (2) CNN의 원리

CNN(Convolutional Neural Network)은 기존 Fully Connected Neural Network와 비교하여 다음과 같은 차별성을 갖습니다.

- 각 레이어의 입출력 데이터의 형상 유지
- 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
- 복수의 필터로 이미지의 특징 추출 및 학습
- 추출한 이미지의 특징을 모으고 강화하는 Pooling 레이어
- 필터를 공유 파라미터로 사용하기 때문에, 일반 인공 신경망과 비교하여 학습 파라미터가 매우 적음

![CNN pic1](https://user-images.githubusercontent.com/117802301/204230714-f7835982-6fed-4e32-a9a9-e133dd34202d.png)


CNN은 위 이미지와 같이 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나눌 수 있습니다. 특징 추출 영역은 Convolution Layer와 Pooling Layer를 여러 겹 쌓는 형태로 구성됩니다. Convolution Layer는 입력 데이터에 필터를 적용 후 활성화 함수를 반영하는 필수 요소입니다. Convolution Layer 다음에 위치하는 Pooling Layer는 선택적인 레이어입니다. CNN 마지막 부분에는 이미지 분류를 위한 Fully Connected 레이어가 추가됩니다. 이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분 사이에 이미지 형태의 데이터를 배열 형태로 만드는 Flatten 레이어가 위치 합니다.

### (3) CNN을 활용한 Tipburn/Healthy 구분
![tipburn_healthy](https://user-images.githubusercontent.com/117802301/204232483-bfc13e25-d6ed-40bd-882f-36da4b9c3e52.png)

우리는 CNN의 딥러닝 기법을 활용하여 각 잎의 픽셀 별로 공간을 유지한채 색상을 파악할 예정입니다. 그리고 색상에 따라 이것이 건강한 잎인지 아닌지를 구별하면서 모델을 구현하고
추가로 이 CNN을 활용했을 때 나타나는 변수들을 설정하고 그에 따른 결과치를 정확도를 판별하여 분석할 예정입니다. 

# Evaluation & Analysis


## 1) Image Crawling
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



## 2) Flower-Recognition-Model



# Result : Flower-Recognition-Model

### (1) Model 1 
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



### (2) Model 2
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
 6) steps_per_epoch = 30 & epochs = 15 

![aix_model2](https://user-images.githubusercontent.com/117802301/204222971-21c23446-bbe8-411e-9d0b-42f038e670d4.png)

#### insight : steps_per_epoch 20 -> 30 & epoch 20 -> 15 : 진동 감소 & 경향성 유지



### (3) Model 3
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
 4) optimizers : RMSprop
 5) lr = 1e-4
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model3](https://user-images.githubusercontent.com/117802301/204223583-f0698d35-01ba-4b38-84fe-02d195e4c1d9.png)

#### insight : data augmentation 을 했더니 진동 증가, 평균 acc도 증가



### (4) Model 4
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
 4) optimizers : RMSprop
 5) lr = 1e-5
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model4](https://user-images.githubusercontent.com/117802301/204223947-47542a27-0a3b-43d1-89f3-c935f5e3d011.png)

#### insight : lr를 1e-4 -> 1e-5 : 진동 증가, acc 감소 = 성능 감소



### (5) Model 5
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
 4) optimizers : RMSprop
 5) lr = 5e-5
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model5](https://user-images.githubusercontent.com/117802301/204223947-47542a27-0a3b-43d1-89f3-c935f5e3d011.png)

#### insight : lr를 1e-5 -> 5e-5 : 진동 증가, acc 감소 = 성능 감소



### (6) Model 6
(1) parameter :
 1) image size : 300 X 300
 2) batch_size : 20
 3) model composition :deeper

```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
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
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model6](https://user-images.githubusercontent.com/117802301/204224379-748586e9-3c9b-4a8c-a2ff-8648f6efaed8.png)

#### insight : lr 다시 1e-4로 & cnn모델 deeper & 이미지 사이즈 300X300 으로 증가, model2와 비교했을 때, cnn 모델을 깊게하고, 이미지 사이즈를 증가시켰을 때 증가하는 경향은 개선됐지만, 진동과 acc 측면에서 아쉬웠음.



### (7) Model 7
(1) parameter :
 1) image size : 300 X 300
 2) batch_size : 20
 3) model composition :

```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
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
 5) lr = 7e-5
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model7](https://user-images.githubusercontent.com/117802301/204225259-9a53c95c-7a86-43c6-bfbc-403190e6e573.png)

#### insight : lr 7e-5 로 바꿨을대, 초반엔 빠른 학습을 했지만 중간에 성능이 나빠지는 경향을 보임. 경향성 자체는 안정적임



### (8) Model 8
(1) parameter :
 1) image size : 300 X 300
 2) batch_size : 20
 3) model composition :

```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
 4) optimizers : Adam
 5) lr = 7e-5
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model8](https://user-images.githubusercontent.com/117802301/204225564-c2a53c5d-84a7-4f51-872b-e037bed7f260.png)

#### insight : optimizer를 adam으로 바꾼 후 과하게 빠른 속도로 학습되는 현상과 경향성, acc 다 좋아졌음.



### (9) Model 9
(1) parameter :
 1) image size : 300 X 300
 2) batch_size : 20
 3) model composition :

```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
 4) optimizers : Adam
 5) lr = 1e-4
 6) steps_per_epoch = 30 & epochs = 15 
 7) Data augmentation 추가

![aix_model9](https://user-images.githubusercontent.com/117802301/204225839-5e0ec504-246e-498e-b580-9cc794053abb.png)

#### insight : model8에서 lr를 7e-5에서 1e-4로 변경하였고, 과적합이 개선되었음. 또한 acc 83%, val_acc 93%로 성능 또한 매우 우수함.


# Conclusion : Discussion

## Best Optimized Model

![model 딥러닝 표](https://user-images.githubusercontent.com/117802301/204227189-844a6992-07ca-46b2-b193-2c8f9ef7f99b.png)
![model 10](https://user-images.githubusercontent.com/117802301/204227347-c7c375aa-6b6c-471d-9c72-57c2d434f2d3.png)

최종적으로 구현한 모델에서 과적함을 극복하고 loss가 감소하면서 accuracy가 증가하는 현상 발생하였습니다.
정확도 93.47%와 val acc가 95%로 모댈의 오차를 최적화한채 구현 하였습니다.


# Related Works
https://goldsystem.tistory.com/822
https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic/report
https://www.kaggle.com/datasets/cf488efb70f71b0db8c5a69539ea35874787d4a4ab835126168e7af1723418d7
- 원영준 교수님 딥러닝 강의자료
