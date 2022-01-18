import pandas as pd

def make_utterances(data_path):
    chatbot_data = pd.read_csv(data_path)
    question = chatbot_data.Q  # Seq2Seq에 encoder_input
    answer = chatbot_data.A  # Seq2Seq에 decoder_input
    total_utterances = pd.concat((question, answer))  # vocab을 만들기위한 전체 발화문장
    return total_utterances, question, answer


def make_vocab(total_utterances, pos_tagger):
    vocab = []
    special_tokens = ['[PAD]', '[MASK]', '[START]', '[END]', '[UNK]']
    for special_token in special_tokens:
        vocab.append(special_token)

    for utterance in total_utterances:
        for eojeols in pos_tagger.pos(utterance, flatten=False, join=True):  # 대화를 형태소 분석기로 나눈 뒤 어절단위로 쪼갠 후
            count = 0
            for token in eojeols:

                if count > 0:
                    if token in vocab:
                        continue
                    vocab.append('##' + token)  # 어절의 뒤에 나오는 형태소에 ##을 붙여 앞의 형태소와 이어지는 형태소임을 명시합니다. ex) 학교에 -> [학교, "##에"]
                else:
                    if token in vocab:
                        continue
                    vocab.append(token)
                    count += 1
    return vocab


