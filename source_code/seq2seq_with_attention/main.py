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
from model import Seq2Seq
from utils import make_total_utterances, make_vocab
from train import Trainer

from konlpy.tag import Mecab  # tweepy오류로 konlpy 직접 설치 필요, mecab별도 설치 readme 참고





def main():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(opt.data_path, opt.batch_size, opt.lr, opt.epochs, device, opt.vocab_size, opt.max_seq_len)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
    parser.add_argument('--max_seq_len', type=int, default=60, help='max sequence length')
    parser.add_argument('--data_path', type=str, default='../../dataset/ChatbotData.csv',
                        help='root directory of the dataset')
    parser.add_argument('--vocab_size', type=str, default=512, help='size of vocab')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    opt = parser.parse_args()


    main(opt)

