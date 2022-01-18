import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
import re
from tqdm.auto import tqdm

from data import WordHandler, ChitChatDataset
from utils import make_utterances, make_vocab
from train import Trainer

from konlpy.tag import Mecab  # tweepy오류로 konlpy 직접 설치 필요, mecab별도 설치 readme 참고



def main(opt):
    total_utterances, question, answer = make_utterances(opt.data_path)
    pos_tagger = Mecab()  # konlpy의 대표적인 형태소 분석기 mecab

    vocab = make_vocab(total_utterances, pos_tagger)

    token2index = {token: index for index, token in enumerate(vocab)}
    index2token = {index: token for index, token in enumerate(vocab)}

    handler = WordHandler(vocab, pos_tagger, token2index, index2token)

    input_ids = question.map(handler.encode)
    output_ids = answer.map(handler.encode)

    chitchat_data = ChitChatDataset(input_ids, output_ids, index2token, token2index, 60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(dataset=chitchat_data, batch_size=opt.batch_size, lr=opt.lr,
                      epoch=opt.epochs, device=device,
                      vocab_size=len(vocab), max_seq_len=opt.max_seq_len, handler=handler)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--max_seq_len', type=int, default=60, help='max sequence length')
    parser.add_argument('--in_dim', type=int, default=512, help='input dimension')
    parser.add_argument('--out_dim', type=int, default=768, help='output dimension')
    parser.add_argument('--data_path', type=str, default='../../dataset/ChatbotData.csv',
                        help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    opt = parser.parse_args()


    main(opt)

