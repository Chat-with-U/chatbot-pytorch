# 💡 Chatbot Pytorch
Pytorch 버전 Open Domain 챗봇 구현 기록 저장소

- NLP를 공부하고자 하는 사람들을 위한 자료 제공
- 데이터 수집, 전처리, 학습, 제공까지 전반적인 과정을 기록

## Chatbot Model
### 1. Seq2Seq
#### 구조
<p align="center">
  <img src="https://user-images.githubusercontent.com/53163222/135714612-e8e4bdcd-e981-4ed8-817b-be0b4fe836c0.png">
  <img src="https://user-images.githubusercontent.com/53163222/135718233-16f07bde-5494-4403-8ccb-044461ac2c43.png">
</p>

Seq2Seq은 크게 Encoder와 Decoder로 이루어져 있다.

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)


## Directory
```
/dataset: 데이터셋
/bert: bert 모델
/seq2seq: seq2seq 모델
```

## Dataset
# 구조
[송영숙님 챗봇 데이터](https://github.com/songys/Chatbot_data) 를 활용<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/53163222/135715869-67949a4c-98d8-45b9-b808-9d4864058661.png">
</p>

- 챗봇 트레이닝용 한글 문답 페어 11,876개 (인공데이터)
- Q(질문), A(답변), label(감정) 세개의 column으로 구성
  - label - 일상: 0, 이별(부정): 1, 사랑(긍정): 2 으로 레이블링
  - Seq2Seq 구조에서는 감정 데이터를 사용하지 않음.
  

## Requirements
- PyTorch 
- CUDA 10.2
- Python 3.7
- Numpy
- Pandas
- KoNLPy
  >#### KoNLPy 설치
  > 데이터 전처리 과정에서 문장을 토큰으로 나눠야하는데, 한국어의 경우 KoNLPy를 활용하여 형태소분석을 한다.
  > KoNLPy를 통해 Hannanum, Kkma, Komoran, Mecab, Okt(Twitter) 등의 라이브러리를 활용할 수 있다.
  > 
  > 
  > **<Windows>**
  > 1.자바 설치(버전 1.7 이상)<br>
  > cmd창에서 버전 확인 `java -version`
  > 
  > 2.JPype1 설치 <br>
  >  파이썬에서 자바를 호출할 수 있는 라이브러리 `pip install jpype1`
  > 
  > 3.KoNLPy 설치 <br>
  > `pip install konlpy`<br>
  > AttributeError: module 'tweepy' has no attribute 'StreamListener' 오류로 [직접 설치](https://github.com/konlpy/konlpy) 해야함.
  > 
  > 참고<br>
  > https://konlpy.org/ko/v0.4.3/install/#id2 <br>
  > https://ericnjennifer.github.io/python_visualization/2018/01/21/PythonVisualization_Chapt1.html

  > ### Mecab 설치 
  > https://hong-yp-ml-records.tistory.com/91
  
## Reference
https://tutorials.pytorch.kr/beginner/chatbot_tutorial.html 를 참고하여 작성.
