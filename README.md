## <p align ="center"> ๐Aiffelthon๐ </p>

## <p align ="center"> ํ๋ก์ ํธ : ์ธ๊ณต์ง๋ฅ ๋น์๋ฅผ ๋ง๋ค์ด๋ณด์! </p> 

### <p align ="center"> ๐ค Team ChatHuman: ๋ฐฉ์น์ฑ ๊ตฌ๋ณธํ ์ดํํ ์ฅ๋ฌธ๊ท ๐ค </p>

### <p align ="center"> 22.12.26 ~ 23.02.08 </p>
---
 
### ์ธ๋ถ ์ผ์ 

|์์|๊ธฐ๊ฐ|์ธ๋ถ ๊ณํ|
|:---:|:---:|:---:|
|1๋ฒ|22.12.26 ~ 22.12.30|ํ์๋ค๊ณผ์ ๊ณํ ์กฐ์จ|
|2๋ฒ|23.01.02 ~ 23.02.03|๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ๋ฐ ๊ฐ๋ฐ ํ๊ฒฝ ๊ตฌ์ถ|
|3๋ฒ|23.01.09 ~ 23.01.13|๋ชจ๋ธ ํ์คํธ|
|4๋ฒ|23.01.09 ~ 23.02.03|์นํ์ด์ง ๊ตฌ์ถ|
|5๋ฒ|23.01.16 ~ 23.01.20|๋ชจ๋ธ ์ฐ๊ตฌ ๋ฐ ์ธํผ๋ฐ์ค ์ฝ๋ ์์ฑ|
|6๋ฒ|23.01.16 ~ 23.01.27|๋ชจ๋ธ ํ์ต|
|7๋ฒ|23.01.18 ~ 23.02.03|๋ชจ๋ธ ๋ค๋ฌ๊ธฐ|
|8๋ฒ|23.02.06 ~ 23.02.07|๋ฐํ ์ค๋น|
|9๋ฒ|23.02.08|๋ฐํ|
 
---
### ๊ฐ์
- ์ผ์์ํ์ ๋์์ ์ฃผ๋ฉฐ ํ์ฌ ์กด์ฌํ๋ ์๋ฆฌ, ํด๋ก๋ฒ, ์นด์นด์ค์ ๊ฐ์ ์ธ๊ณต์ง๋ฅ ๋น์์๊ฒ ๋ถ์ฌ๋ ๊ธฐ๋ฅ์ ํ์ฌํ๋๊ฒ์ ๋ชฉํ๋ก ํจ.
  - ๋ถ์ฌ๋ ๊ธฐ๋ฅ์ด๋ผํจ์ ์ผ์๋ํ ๊ธฐ๋ฅ์ ์ด์ ์ ๋ง์ถ๋ฉฐ ๊ทธ ์ธ์ ์ฌ์ํ ๊ฒ๋ค์ ์ถ๊ฐํด๋ณผ ์์ .   
  
- ๊ธฐ๋ณธ์ ์ธ ํํ๋ ์ฑ๋ด ํํ์ด๋ฉฐ ๋ค์ํ ํค์๋๋ฅผ ํตํด ๋ช๋ น์ ์ํํ  ์ ์๋๋ก ํ  ๊ฒ.

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
- AIHub ์์ ์ ๊ณตํ๋ ์ฃผ์ ๋ณ ํ์คํธ ์ผ์์ํ ๋ฐ์ดํฐ์ ํ๊ตญ์ด ๋ํ ์์ฝ์ ์ด์ฉํ์ฌ ๋ง๋ฆ.
- ๋ฐ์ดํฐํ๋ ์ ํํ๋ก ํ ๋ํ์ ๋ง๋ญ์น๋ฅผ Conversation column์ผ๋ก ๊ตฌ๋ถํ๊ณ  ๊ฐ ๋ํ ๊ฐ ๋ฐํ์๋ฅผ `'<usr>'`, `'<sys>'` ํ ํฐ์ผ๋ก ๊ตฌ๋ถํ์์.
- ์ฝ 19๋ง๊ฐ์ ๋ํ๋ฅผ ์ด์ฉํจ. 

![data_image](https://github.com/Ukbang/Aiffel_thon/blob/main/images/data_image.jpeg)

---
### ์ ์ฒ๋ฆฌ ๋ฐฉ์
- modules/preprocessing.py ํ์ผ์ clear_sentence ํจ์๋ฅผ ์ด์ฉํ์ฌ ์ฒ๋ฆฌ.
- #@์ด๋ฆ#์ make_name ํจ์๋ฅผ ์ด์ฉํ์ฌ ๋๋คํ ์ด๋ฆ์ ์์ฑํ  ์ ์๋๋ก ํ์์.
- @URL, #@์์คํ#์ฌ์ง#, #@์ด๋ชจํฐ์ฝ#์ ์ญ์ ํ์๊ณ  ๋ฐ๋ณต๋๋ ใ,ใ,ใ,ใ ,. ๊ณผ ๊ฐ์ ๋ฌธ์๋ 2๊ฐ๋ก ํต์ผํ์์ผ๋ฉฐ ์์ฃผ ๋ฑ์ฅํ๋ ํคํค ๋ ใใ ๋ก ๋ณ๊ฒฝํ์์.

![make_name](https://github.com/Ukbang/Aiffel_thon/blob/main/images/make_name.jpeg)
![clear_sentence](https://github.com/Ukbang/Aiffel_thon/blob/main/images/clear_sentence.jpeg)

---

### ๋ชจ๋ธ
- ๋ชจ๋ธ์ ๐คHugging Face์์ ์ ๊ณตํ๋ gpt2 ๋ชจ๋ธ์ ์ฌ์ฉํ์์.
- ๋ฒ ์ด์ค ๋ชจ๋ธ๋ก ['skt/kogpt2-base-v2'](https://github.com/SKT-AI/KoGPT2) ์ ์ฌ์ฉํจ.   
 
 
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884750-53fb4373-2d9d-4a6a-800b-0062b8b702f5.png" width="800px" height="300px"></p>

---
### ํ์ต ์งํ๊ณผ์  ๋ฆฌ๋๋ณด๋
#### Data type
__Topic = 250000๊ฐ__
 
 __Topic+kakao Data = 190000๊ฐ ('`<usr>`๋ก ๋๋๋ ๋ฌธ๊ตฌ ์ญ์ ', ๊ธธ์ด 256')__
 
__kakao Data = 65000๊ฐ__

---

#### Label

__Input = Input๊ณผ Label์ด ๋์ผ__
 
 
__-100 = ๋ง์ง๋ง `<sys>` ๋ํ๋ฅผ ์ ์ธํ -100์ ์ด์ฉํ Masking__

__-100+sys = ๋ชจ๋  `<sys>` ๋ํ๋ฅผ ์ ์ธํ ๋ชจ๋  ๋ํ -100์ผ๋ก Masking__
 
 
__Shift = Input์ `<s>` ํ ํฐ์ bos_token์ผ๋ก ์ฌ์ฉ, Label์ `</s>`ํ ํฐ์ eos_token์ผ๋ก ์ฌ์ฉํจ.__

|index|Model|Epochs|Data type|์งํ ์ํฉ|์งํ ์ผ์|Label|Loss|Val_Loss|Comment|์ฑ๋ฅ|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-01-31|-100|4.290 -> 3.797 -> 3.340 -> 2.803 -> 2.195|3.821 -> 3.759 -> 3.804 -> 3.938 -> 4.143|๋จ๋ตํ์ด๊ณ  ๋ํ๊ฐ ์ ์ด๋ฃจ์ด ์ง์ง ์์.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_-100_test.ipynb)|
|2|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|Input|1.476 -> 1.343 -> 1.270 -> 1.203 -> 1.137|1.486 -> 1.445 -> 1.434 -> 1.441 -> 1.461|ํ์ฌ๊น์ง ๊ฐ์ฅ Best|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/23-02-01_True_test.ipynb)|
|3|skt/kogpt2-base-v2|3|kakao Data|Done|2023-01-30|Input|2.330 -> 2.147 -> 2.084|1.765 -> 1.723 -> 1.704|๋ฌธ์ฅ ์์ฑ์ eos token ๋ฐ์ ๋ชปํจ.|[Link](https://github.com/Ukbang/Aiffel_thon/blob/main/chatbot/Test/Inference_code_label_True_len384.ipynb)|
|4|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-01|shift|2.140 -> 2.005 -> 1.931 -> 1.864 -> 1.794|2.298 -> 2.236 -> 2.215 -> 2.221 -> 2.246|ํ์ต์ด ์ ํ ๋์ง ์์์. ํ๊ธฐ|[Link]()|
|5|skt/kogpt2-base-v2|10|Topic+kakao|Done|2023-02-01|Input|1.483 -> 1.352 -> 1.275 -> 1.206 -> 1.135 -> 1.062 -> 0.986 -> 0.908 -> 0.830 -> 0.753|1.504 -> 1.469 -> 1.456 -> 1.463 -> 1.485 -> 1.517 -> 1.562 -> 1.616 -> 1.683 -> 1.759|5epoch ์ด์๋ถํฐ ํ์ต์ด ์คํ๋ ค ์๋จ. ํ๊ธฐ|[Link]()|
|6|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|Input|1.479 -> 1.380 -> 1.330 -> 1.292 -> 1.260|1.401 -> 1.370 -> 1.357 -> 1.350 -> 1.346|์์ํ ์ฝ๋ ์์ ํ ํ์ตํ์์. 2๋ฒ๊ณผ ์ฑ๋ฅ์ด ๋์ผํจ.|[Link]()|
|7|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100|4.269 -> 3.869 -> 3.574 -> 3.296 -> 3.031|4.069 -> 3.997 -> 3.989 -> 4.032 -> 4.106|.....|[Link]()|
|8|skt/kogpt2-base-v2|5|Topic+kakao|Done|2023-02-05|-100+sys|3.826 -> |3.352 -> |.....|[Link]()|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

---

### ์ถ๊ฐ๋ ์๋น์ค
<p align ="center"><img src="https://user-images.githubusercontent.com/112140135/216884826-5905e7cb-229a-4a53-becd-25508e40fd1d.png" width="600px" height="900px"></p>

---

### ํ๊ณ 
#### ํ๋ก์ ํธ ์ํ์ ์ด๋ ค์ ๊ทน๋ณต ๊ฒฝํ
  - 
  - 
 
#### ํ๋ก์ ํธ์์ ์ํ ๊ฒฝํ
  - 
  - 
  
#### ํ๋ก์ ํธ์์ ์์ฌ์ด ๊ฒฝํ
  - 
  - 

---
### ์ฐธ๊ณ  ๋ฌธํ
1. [songys/AwesomeKorean_Data: ํ๊ตญ์ด ๋ฐ์ดํฐ ์ธํธ ๋งํฌ](https://github.com/songys/AwesomeKorean_Data)
2. [์์ ๋ํํ์์ ์์ฑ ๋ฐ์ดํฐ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=109)
3. [STT๋ชจ๋ธ ๋ฐ TTS๋ชจ๋ธ ๊ฐ๋ฐ](https://www.youtube.com/watch?v=WTul6LIjIBA)
4. [์จ๋ผ์ธ ๊ตฌ์ด์ฒด ๋ง๋ญ์น ๋ฐ์ดํฐ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625)
5. [๋ฒ๋ฅ  ์ง์ ๋ฒ ์ด์ค](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=99)
6. [ํ์ด์ฌ์ผ๋ก JSON ํ์ผ ๋ค๋ฃจ๊ธฐ](https://www.youtube.com/watch?v=s9D-JIuaFqY&t=433s)
7. [korean SmileStyle Dataset](https://www.google.com/url?q=https://github.com/smilegate-ai/korean_smile_style_dataset&sa=D&source=docs&ust=1672048006339662&usg=AOvVaw2KWZl71R1gdPiznFcT1tkG)
8. [์ฃผ์ ๋ณ ํ์คํธ ์ผ์์ํ ๋ฐ์ดํฐ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=543)
9. [ํ๊ตญ์ด ๋ํ ์์ฝ](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=117)
10. [ํ๊นํ์ด์ค ๋ชจ๋ธ](https://huggingface.co/lcw99/ko-dialoGPT-korean-chit-chat)
11. [[NLP] ์ธ์ด๋ชจ๋ธ์ ํ๊ฐ์งํ 'Perplexity' ๊ฐ๋ ๋ฐ ๊ณ์ฐ๋ฐฉ๋ฒ](https://heytech.tistory.com/344)
12. [๋ฌด์จ ๋ํ๋  ํ  ์ ์๋ ์์ด์ ํธ๋ฅผ ํฅํ์ฌ](https://brunch.co.kr/@synabreu/35)
13. [PyTorch 2.0 ๋ฌด์์ด ๋ค๋ฅธ๊ฐ?](https://blog.naver.com/october-eight/222948663006)
14. [Tensorflow_KoGPT2_Chabot](https://github.com/ukairia777/tensorflow-kogpt2-chatbot/blob/main/KoGPT2_Chatbot.ipynb)
15. [GPT-2 Fine Tuning ](https://blog.naver.com/ds_penaut/222699897818)
16. [CaFeCoKe/KoGPT2_Chatbot](https://github.com/CaFeCoKe/KoGPT2_Chatbot)

---
### ํ์ ๊นํ๋ธ ๋งํฌ

- [๋ฐฉ์น์ฑ](https://github.com/Ukbang)
- [๊ตฌ๋ณธํ](https://github.com/HughBGrant) 
- [์ดํํ](https://github.com/leetaehwan) 
- [์ฅ๋ฌธ๊ท](https://github.com/MunGyuJang)
