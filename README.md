## <p align ="center"> 🏃Aiffelthon🏃 </p>

## <p align ="center"> 프로젝트 : 인공지능 비서를 만들어보자! </p> 

### <p align ="center"> 🤖 Team ChatHuman: 방승욱 구본회 이태환 장문규 🤖 </p>

### <p align ="center"> 22.12.26 ~ 23.02.08 </p>
---
 
### 세부 일정

|순서|기간|세부 계획|
|:---:|:---:|:---:|
|1번|22.12.26 ~ 22.12.30|팀원들과의 계획 조율|
|2번|23.01.02 ~ 23.02.03|데이터 전처리 및 개발 환경 구축|
|3번|23.01.09 ~ 23.01.13|모델 테스트|
|4번|23.01.09 ~ 23.02.03|웹페이지 구축|
|5번|23.01.16 ~ 23.01.20|모델 연구 및 인퍼런스 코드 작성|
|6번|23.01.16 ~ 23.01.27|모델 학습|
|7번|23.01.18 ~ 23.02.03|모델 다듬기|
|8번|23.02.06 ~ 23.02.07|발표 준비|
|9번|23.02.08|발표|
 
---
### 개요
![제목 없음](https://user-images.githubusercontent.com/112141259/228016039-8c5d3284-b4df-4b9f-bc6b-23875f40a6fd.png)

- 일상생활에 도움을 주며 현재 존재하는 시리, 클로버, 카카오와 같은 인공지능 비서에게 부재된 기능을 탑재하는것을 목표로 함.
  - 부재된 기능이라함은 일상대화 기능에 초점을 맞추며 그 외의 사소한 것들을 추가해볼 예정.   
  
- 기본적인 형태는 챗봇 형태이며 다양한 키워드를 통해 명령을 수행할 수 있도록 할 것.

![service](https://github.com/Ukbang/Aiffel_thon/blob/main/images/service.png)

---

### Requirement
> Python 3.9.7
> 
> Transformer 4.11.3
>  
> Numpy 1.21.4
>  
> PyTorch 1.9.1+cu111

---

### Dataset
- AIHub 에서 제공하는 주제별 텍스트 일상생활 데이터와 한국어 대화 요약을 이용하여 만듦.
- 데이터프레임 형태로 한 대화의 말뭉치를 Conversation column으로 구분하고 각 대화 간 발화자를 `'<usr>'`, `'<sys>'` 토큰으로 구분하였음.
- 약 19만개의 대화를 이용함. 

![data_image](https://github.com/Ukbang/Aiffel_thon/blob/main/images/data_image.jpeg)

---
### 전처리 방식
- modules/preprocessing.py 파일의 clear_sentence 함수를 이용하여 처리.
- #@이름#은 make_name 함수를 이용하여 랜덤한 이름을 생성할 수 있도록 하였음.
- @URL, #@시스템#사진#, #@이모티콘#은 삭제하였고 반복되는 ㅋ,ㅎ,ㅜ,ㅠ,. 과 같은 문자는 2개로 통일하였으며 자주 등장하는 키키 는 ㅋㅋ 로 변경하였음.

![make_name](https://github.com/Ukbang/Aiffel_thon/blob/main/images/make_name.jpeg)
![clear_sentence](https://github.com/Ukbang/Aiffel_thon/blob/main/images/clear_sentence.jpeg)

---

### 모델
- 모델은 🤗Hugging Face에서 제공하는 gpt2 모델을 사용하였음.
- 베이스 모델로 ['skt/kogpt2-base-v2'](https://github.com/SKT-AI/KoGPT2) 을 사용함.   
 
 
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884750-53fb4373-2d9d-4a6a-800b-0062b8b702f5.png" width="800px" height="300px"></p>

---
### 학습 진행과정 리더보드
#### Data type
__Topic = 250000개__
 
 __Topic+kakao Data = 190000개 ('`<usr>`로 끝나는 문구 삭제', 길이 256')__
 
__kakao Data = 65000개__

---

#### Label

__Input = Input과 Label이 동일__
 
 
__-100 = 마지막 `<sys>` 대화를 제외한 -100을 이용한 Masking__

__-100+sys = 모든 `<sys>` 대화를 제외한 모든 대화 -100으로 Masking__
 
 
__Shift = Input은 `<s>` 토큰을 bos_token으로 사용, Label은 `</s>`토큰을 eos_token으로 사용함.__

|index|Model|Epochs|Data type|진행 상황|진행 일시|Label|Loss|Val_Loss|Comment|성능|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-01-31|-100|4.290 -> 3.797 -> 3.340 -> 2.803 -> 2.195|3.821 -> 3.759 -> 3.804 -> 3.938 -> 4.143|단답형이고 대화가 잘 이루어 지지 않음.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_-100_test.ipynb)|
|2|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|Input|1.476 -> 1.343 -> 1.270 -> 1.203 -> 1.137|1.486 -> 1.445 -> 1.434 -> 1.441 -> 1.461|현재까지 가장 Best|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_True_test.ipynb)|
|3|skt/kogpt2-base-v2|3|kakao Data|Done|2023-01-30|Input|2.330 -> 2.147 -> 2.084|1.765 -> 1.723 -> 1.704|문장 생성을 eos token 밖에 못함.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/Inference_code_label_True_len384.ipynb)|
|4|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|shift|2.140 -> 2.005 -> 1.931 -> 1.864 -> 1.794|2.298 -> 2.236 -> 2.215 -> 2.221 -> 2.246|학습이 전혀 되지 않았음. 폐기|[Link]()|
|5|skt/kogpt2-base-v2|10|Topic+kakao|Done|2023-02-01|Input|1.483 -> 1.352 -> 1.275 -> 1.206 -> 1.135 -> 1.062 -> 0.986 -> 0.908 -> 0.830 -> 0.753|1.504 -> 1.469 -> 1.456 -> 1.463 -> 1.485 -> 1.517 -> 1.562 -> 1.616 -> 1.683 -> 1.759|5epoch 이상부터 학습이 오히려 안됨. 폐기|[Link]()|
|6|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|Input|1.479 -> 1.380 -> 1.330 -> 1.292 -> 1.260|1.401 -> 1.370 -> 1.357 -> 1.350 -> 1.346|자잘한 코드 수정후 학습하였음. 2번과 성능이 동일함.|[Link]()|
|7|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100|4.269 -> 3.869 -> 3.574 -> 3.296 -> 3.031|4.069 -> 3.997 -> 3.989 -> 4.032 -> 4.106|.....|[Link]()|
|8|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100+sys|3.826 -> |3.352 -> |.....|[Link]()|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

---

### 추가된 서비스
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884826-5905e7cb-229a-4a53-becd-25508e40fd1d.png" width="600px" height="900px"></p>

---

### 회고
#### 프로젝트 수행상 어려움 극복 사례
  - csv로 된 데이터를 학습에 들어갈 데이터 형태로 처리하는 과정이 많이 어려웠던 것 같습니다. 레퍼런스들을 참고해본 결과 token_type_ids는 usr, sys token ids로 채워넣고 labels는 시스템의 마지막 대화만 남기고 임의의 mask 토큰(-100)으로 바꾸거나 input_ids를 그대로 넣는 방식 등 여러가지가 있을 수 있다는 사실을 알게 되었습니다.
 
#### 프로젝트에서 잘한 부분
  - 팀원들과 함께 만든 train 코드를 전반적으로 살펴보면서 필요없는 코드, 시간복잡도가 높은 코드 등을 수정하면서 좀 더 빠르고 깔끔한 코드를 만들 수 있었습니다.
  
#### 프로젝트에서 아쉬운 부분
  - 참고한 레퍼런스들에서는 데이터셋의 labels를 대부분 mask 토큰(-100)으로 가리는 방식을 사용했는데 막상 성능이 안나와서 labels를 input_ids로 지정한 데이터가 더 성능이 잘나오는 점에 대해서 아직 이유를 파악하지 못한 것이 아쉽습니다.

---
### 참고 문헌
1. [songys/AwesomeKorean_Data: 한국어 데이터 세트 링크](https://github.com/songys/AwesomeKorean_Data)
2. [자유대화형식의 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=109)
3. [STT모델 및 TTS모델 개발](https://www.youtube.com/watch?v=WTul6LIjIBA)
4. [온라인 구어체 말뭉치 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625)
5. [법률 지식 베이스](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=99)
6. [파이썬으로 JSON 파일 다루기](https://www.youtube.com/watch?v=s9D-JIuaFqY&t=433s)
7. [korean SmileStyle Dataset](https://www.google.com/url?q=https://github.com/smilegate-ai/korean_smile_style_dataset&sa=D&source=docs&ust=1672048006339662&usg=AOvVaw2KWZl71R1gdPiznFcT1tkG)
8. [주제별 텍스트 일상생활 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=543)
9. [한국어 대화 요약](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117)
10. [허깅페이스 모델](https://huggingface.co/lcw99/ko-dialoGPT-korean-chit-chat)
11. [[NLP] 언어모델의 평가지표 'Perplexity' 개념 및 계산방법](https://heytech.tistory.com/344)
12. [무슨 대화든 할 수 있는 에이전트를 향하여](https://brunch.co.kr/@synabreu/35)
13. [PyTorch 2.0 무엇이 다른가?](https://blog.naver.com/october-eight/222948663006)
14. [Tensorflow_KoGPT2_Chabot](https://github.com/ukairia777/tensorflow-kogpt2-chatbot/blob/main/KoGPT2_Chatbot.ipynb)
15. [GPT-2 Fine Tuning ](https://blog.naver.com/ds_penaut/222699897818)
16. [CaFeCoKe/KoGPT2_Chatbot](https://github.com/CaFeCoKe/KoGPT2_Chatbot)

---
### 팀원 깃허브 링크

- [방승욱](https://github.com/Ukbang)
- [구본회](https://github.com/HughBGrant) 
- [이태환](https://github.com/leetaehwan) 
- [장문규](https://github.com/MunGyuJang)