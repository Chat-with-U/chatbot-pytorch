import torch
from torch import nn

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True, dropout=0.3)

    def forward(self, encoder_embeds):
        encoder_output, hidden_and_cell = self.lstm(encoder_embeds)

        return encoder_output, hidden_and_cell


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, use_attention):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True, dropout=0.3)
        self.softmax_score = nn.Softmax(-1)
        self.use_attention = use_attention

    def forward(self, decoder_embeds, hidden_and_cell, encoder_output=None):

        time_step = decoder_embeds.size()[1]
        decoder_lstm_output = torch.zeros_like(decoder_embeds)  # [Batch, max_seq_len, hidden_dim]\

        if self.use_attention:
            attention_representation = torch.zeros_like(encoder_output)  # [Batch, max_seq_len, hidden_dim]
            transposed_encode_hidden = torch.transpose(encoder_output, 1, 2)  # [Batch, hidden_dim, max_seq_len]

            for step, i in enumerate(range(time_step)):

                if step == 0:
                    decoder_vector_step, h_and_c = self.lstm(decoder_embeds[:, i, :].unsqueeze(1),
                                                             hidden_and_cell)  # [Batch, 1, hidden_dim]
                else:
                    decoder_vector_step, h_and_c = self.lstm(decoder_vector_step, h_and_c)

                softmax_attention = self.softmax_score(
                    torch.bmm(decoder_vector_step, transposed_encode_hidden))  # [Batch, 1, max_seq_len]
                attention_dim = torch.bmm(softmax_attention, encoder_output)  # [Bach, 1, hidden]
                attention_representation[:, i, :] = attention_dim.squeeze(1)
                decoder_lstm_output[:, i, :] = decoder_vector_step.squeeze(1)

            output = torch.cat([decoder_lstm_output, attention_representation], -1)

        else:
            for step, i in enumerate(range(time_step)):

                if step == 0:
                    decoder_vector_step, h_and_c = self.lstm(decoder_embeds[:, i, :].unsqueeze(1),
                                                             hidden_and_cell)  # [Batch, 1, hidden_dim]
                else:
                    decoder_vector_step, h_and_c = self.lstm(decoder_vector_step, h_and_c)

                decoder_lstm_output[:, i, :] = decoder_vector_step.squeeze(1)
            output = decoder_lstm_output

        return output



class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, use_attention=None, is_test=None):
        super(Seq2Seq, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = LSTMEncoder(embedding_dim)
        self.decoder = LSTMDecoder(embedding_dim, use_attention)
        self.vocab_proj = nn.Linear(embedding_dim, vocab_size)

        if use_attention:
            self.vocab_proj = nn.Linear(embedding_dim * 2, vocab_size)
        self.use_attention = use_attention

    def forward(self, encode_input, decode_input):
        encoder_embeds = self.word_embedding(encode_input)
        decoder_embeds = self.word_embedding(decode_input)

        encoder_output, hidden_and_cell = self.encoder(encoder_embeds)
        decoder_output = self.decoder(decoder_embeds, hidden_and_cell, encoder_output)

        projected_output = self.vocab_proj(decoder_output)

        return projected_output

