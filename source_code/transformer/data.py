from torch.utils.data import Dataset
import torch

class WordHandler:
    def __init__(self, vocab, pos_tagger, token2index, index2token):
        self.vocab = vocab
        self.pos_tagger = pos_tagger
        self.token2index = token2index
        self.index2token = index2token

    def encode(self, sentence):
        encoded_vector = [self.token2index[token] if token in self.token2index else self.token2index['[UNK]']
                          for token in self.pos_tagger.pos(sentence, join=True)]

        return encoded_vector

    def decode(self, indice, join=True):
        decoded_vector = [self.index2token[index] for index in indice]

        return decoded_vector

    def decode_without_tag(self, indice):
        decoded_vector = ' '.join([self.index2token[index].split('/')[0] for index in indice])

        return decoded_vector

    @staticmethod
    def return_max_seq_len(sentences):
        max_seq_len = 0
        for sentence in sentences:
            max_seq_len = max(len(sentence), max_seq_len)

        return max_seq_len

        # 이 handler는 주어진 데이터셋에서 가장 긴 문장길이를 max_seq_len로 return합니다.
    # 따라서 긴 길이의 문장이 잘려 손실이 발생하진 않지만 짧은 문장은 거의 [PAD]토큰으로 채워질 수 있습니다.


class ChitChatDataset(Dataset):
    def __init__(self, input_ids, output_ids, index2token, token2index, max_seq_len):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.index2token = index2token
        self.token2index = token2index
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):

        if len(self.input_ids[idx]) + 2 < self.max_seq_len:
            padding_block = self.max_seq_len - len(self.input_ids[idx]) + 2
            input = torch.LongTensor([self.token2index['[START]']] +
                                     self.input_ids[idx] +
                                     [self.token2index['[END]']] +
                                     [self.token2index['[PAD]']] * padding_block)
        else:
            input = torch.LongTensor([self.token2index['[START]']] +
                                     self.input_ids[idx] +
                                     [self.token2index['[END]']])

        if len(self.output_ids[idx]) + 2 < self.max_seq_len:
            padding_block = self.max_seq_len - len(self.output_ids[idx]) + 2
            output = torch.LongTensor([self.token2index['[START]']] +
                                      self.output_ids[idx] +
                                      [self.token2index['[END]']] +
                                      [self.token2index['[PAD]']] * padding_block)
        else:
            output = torch.LongTensor([self.token2index['[START]']] +
                                      self.output_ids[idx] +
                                      [self.token2index['[END]']])

        return input, output

    def __len__(self):
        return len(self.input_ids)

