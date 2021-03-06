# ๐ก Chatbot Pytorch
Pytorch ๋ฒ์  Open Domain ์ฑ๋ด ๊ตฌํ ๊ธฐ๋ก ์ ์ฅ์

- NLP๋ฅผ ๊ณต๋ถํ๊ณ ์ ํ๋ ์ฌ๋๋ค์ ์ํ ์๋ฃ ์ ๊ณต
- ๋ฐ์ดํฐ ์์ง, ์ ์ฒ๋ฆฌ, ํ์ต, ์ ๊ณต๊น์ง ์ ๋ฐ์ ์ธ ๊ณผ์ ์ ๊ธฐ๋ก

## Chatbot Model
### 1. Seq2Seq
#### ๊ตฌ์กฐ
<p align="center">
  <img src="https://user-images.githubusercontent.com/53163222/135714612-e8e4bdcd-e981-4ed8-817b-be0b4fe836c0.png">
  <img src="https://user-images.githubusercontent.com/53163222/135718233-16f07bde-5494-4403-8ccb-044461ac2c43.png">
</p>

Seq2Seq์ ํฌ๊ฒ Encoder์ Decoder๋ก ์ด๋ฃจ์ด์ ธ ์๋ค.

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)


## Directory
```
/dataset: ๋ฐ์ดํฐ์
/bert: bert ๋ชจ๋ธ
/seq2seq: seq2seq ๋ชจ๋ธ
```

## Dataset
# ๊ตฌ์กฐ
[์ก์์๋ ์ฑ๋ด ๋ฐ์ดํฐ](https://github.com/songys/Chatbot_data) ๋ฅผ ํ์ฉ<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/53163222/135715869-67949a4c-98d8-45b9-b808-9d4864058661.png">
</p>

- ์ฑ๋ด ํธ๋ ์ด๋์ฉ ํ๊ธ ๋ฌธ๋ต ํ์ด 11,876๊ฐ (์ธ๊ณต๋ฐ์ดํฐ)
- Q(์ง๋ฌธ), A(๋ต๋ณ), label(๊ฐ์ ) ์ธ๊ฐ์ column์ผ๋ก ๊ตฌ์ฑ
  - label - ์ผ์: 0, ์ด๋ณ(๋ถ์ ): 1, ์ฌ๋(๊ธ์ ): 2 ์ผ๋ก ๋ ์ด๋ธ๋ง
  - Seq2Seq ๊ตฌ์กฐ์์๋ ๊ฐ์  ๋ฐ์ดํฐ๋ฅผ ์ฌ์ฉํ์ง ์์.
  

## Requirements
- PyTorch 
- CUDA 10.2
- Python 3.7
- Numpy
- Pandas
- KoNLPy
  >#### KoNLPy ์ค์น
  > ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ๊ณผ์ ์์ ๋ฌธ์ฅ์ ํ ํฐ์ผ๋ก ๋๋ ์ผํ๋๋ฐ, ํ๊ตญ์ด์ ๊ฒฝ์ฐ KoNLPy๋ฅผ ํ์ฉํ์ฌ ํํ์๋ถ์์ ํ๋ค.
  > KoNLPy๋ฅผ ํตํด Hannanum, Kkma, Komoran, Mecab, Okt(Twitter) ๋ฑ์ ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ํ์ฉํ  ์ ์๋ค.
  > 
  > 
  > **<Windows>**
  > 1.์๋ฐ ์ค์น(๋ฒ์  1.7 ์ด์)<br>
  > cmd์ฐฝ์์ ๋ฒ์  ํ์ธ `java -version`
  > 
  > 2.JPype1 ์ค์น <br>
  >  ํ์ด์ฌ์์ ์๋ฐ๋ฅผ ํธ์ถํ  ์ ์๋ ๋ผ์ด๋ธ๋ฌ๋ฆฌ `pip install jpype1`
  > 
  > 3.KoNLPy ์ค์น <br>
  > `pip install konlpy`<br>
  > AttributeError: module 'tweepy' has no attribute 'StreamListener' ์ค๋ฅ๋ก [์ง์  ์ค์น](https://github.com/konlpy/konlpy) ํด์ผํจ.
  > 
  > ์ฐธ๊ณ <br>
  > https://konlpy.org/ko/v0.4.3/install/#id2 <br>
  > https://ericnjennifer.github.io/python_visualization/2018/01/21/PythonVisualization_Chapt1.html

  > ### Mecab ์ค์น 
  > https://hong-yp-ml-records.tistory.com/91
  
## Reference
https://tutorials.pytorch.kr/beginner/chatbot_tutorial.html ๋ฅผ ์ฐธ๊ณ ํ์ฌ ์์ฑ.
