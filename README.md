# 💡 Chatbot Study
Open Domain 챗봇 구현을 위한 스터디 내용 기록 저장소

- NLP를 공부하고자 하는 사람들을 위한 자료 제공
- 데이터 수집, 전처리, 학습, 제공까지 전반적인 과정을 기록

## Chatbot Model
### 1. Bert
#### 구조
<p align="center">
  <img src="https://user-images.githubusercontent.com/53163222/135712814-34333b78-24d1-42b9-8811-56931720edcc.png">
  <img src="https://user-images.githubusercontent.com/53163222/135712806-4b064e04-e560-4768-99f1-a9b52fb926b5.png">
</p>

#### 특징
#### 장점
#### 단점

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### 2. Seq2Seq
#### 구조
<p align="center">
  <img src="https://user-images.githubusercontent.com/53163222/135714612-e8e4bdcd-e981-4ed8-817b-be0b4fe836c0.png">
  <img src="https://user-images.githubusercontent.com/53163222/135718233-16f07bde-5494-4403-8ccb-044461ac2c43.png">
</p>

Seq2Seq은 크게 Encoder와 Decoder로 이루어져 있다.


#### 특징
#### 장점
#### 단점

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
- Python 3.7
- Numpy
-


## Reference