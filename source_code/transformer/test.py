from transformers import AutoTokenizer, AutoModel
from models import Encoder, Embedding, Transformer

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

sample_input = tokenizer.encode('이순신은 조선 중기의 무신이다',
                 add_special_tokens=True,
                 padding='max_length',
                 max_length=100,
                 truncation=True,
                 return_tensors="pt")

sample_output = tokenizer.encode('이순신은 조선 중기의 무신이다',
                 add_special_tokens=True,
                 padding='max_length',
                 max_length=100,
                 truncation=True,
                 return_tensors="pt")

embdding = Embedding(8002, 512)
encoder = Encoder(512, 768, 4)

model = Transformer(8002, 1, 6, 512, 768)

output = model(sample_input, sample_output)

a = 0
