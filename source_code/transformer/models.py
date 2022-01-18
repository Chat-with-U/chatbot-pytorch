import torch
from torch import nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocab_size, in_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, in_dim)

    def forward(self, word_vector):
        return self.embedding(word_vector)


class ScaledDotProduct(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self, out_dim):
        super(ScaledDotProduct, self).__init__()
        self.scale_attn_table = nn.Softmax(-1)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, query, key, value, mask, d_k):
        device = query.device
        if mask == None:
            attention_table = ((query.matmul(key.transpose(2, 3)) / np.sqrt(d_k)))
        else:
            attention_table = ((query.matmul(key.transpose(2, 3)) / np.sqrt(d_k)) * mask)
            attention_table = torch.where(attention_table > 0, attention_table, torch.FloatTensor([-10000]).to(device))

        attention_score = self.scale_attn_table(attention_table)
        attention_out = attention_score.matmul(value)

        return attention_out


class MultiHeadAttention(nn.Module):
    """
    torch Model Snippet
    """

    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.query = nn.Linear(in_dim, num_heads * out_dim)
        self.key = nn.Linear(in_dim, num_heads * out_dim)
        self.value = nn.Linear(in_dim, num_heads * out_dim)
        self.scaled_dot_product = ScaledDotProduct(out_dim)
        self.word_dim_projection = nn.Linear(num_heads * out_dim, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)

        self.num_heads = num_heads

    def forward(self, q, k, v, mask=None):
        residual = q
        batch, q_seq_len, q_word_dim = q.size()
        _, kv_seq_len, kv_word_dim = k.size()
        device = q.device

        query = self.query(q).view(batch,
                                   q_seq_len,
                                   self.num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, word_dim]
        key = self.key(k).view(batch,
                               kv_seq_len,
                               self.num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, word_dim]
        value = self.value(v, ).view(batch,
                                     kv_seq_len,
                                     self.num_heads, -1).transpose(1, 2)  # [batch, num_heads, seq_len, word_dim]
        if mask is not None:
            mask = mask.repeat(1, self.num_heads, 1, 1).to(device)
        d_k = query.size()[-1]
        attention_out = self.scaled_dot_product(query, key, value, mask, d_k)
        contiguous = attention_out.transpose(1, 2).reshape(batch, q_seq_len, -1)
        output = self.word_dim_projection(contiguous)

        return self.layer_norm(residual + output)


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.forward_net = nn.Linear(in_dim, out_dim)
        self.recall_net = nn.Linear(out_dim, in_dim)
        self.relu = nn.ReLU()

    def forward(self, sequence):
        forward = self.relu(self.forward_net(sequence))
        recall = self.relu(self.recall_net(forward))
        return recall


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(in_dim, out_dim, num_heads)
        self.feed_forward = FeedForward(in_dim, out_dim)

    def forward(self, encoder_embedding):
        after_attention_vector = self.attention(encoder_embedding,
                                                encoder_embedding,
                                                encoder_embedding)
        encoder_output = self.feed_forward(after_attention_vector)
        return encoder_output


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, teacher_forcing_rate=0.3):
        super(Decoder, self).__init__()
        self.self_attention = MultiHeadAttention(in_dim, out_dim, num_heads)
        self.self_forward = FeedForward(in_dim, out_dim)

        self.encoder_decoder_attention = MultiHeadAttention(in_dim, out_dim, num_heads)
        self.encoder_decoder_forward = FeedForward(in_dim, out_dim)

        self.teacher_forcing_rate = teacher_forcing_rate

    def forward(self, encoder_output, decoder_embedding):
        batch_size, seq_len, in_dim = decoder_embedding.size()
        output = torch.zeros_like(decoder_embedding)

        for seq_order in range(1, seq_len + 1):
            mask = torch.ones(batch_size, 1, seq_order, seq_order).tril()
            self_attention_input = decoder_embedding[:, :seq_order, :]
            self_attention_output = self.self_attention(self_attention_input,
                                                        self_attention_input,
                                                        self_attention_input,
                                                        mask)

            feed_forward_output = self.self_forward(self_attention_output)

            encoder_decoder_att_output = self.encoder_decoder_attention(feed_forward_output,
                                                                        encoder_output,
                                                                        encoder_output,
                                                                        )
            feed_forward_output = self.encoder_decoder_forward(encoder_decoder_att_output)

            if torch.rand(1) < self.teacher_forcing_rate:
                continue
            else:
                if seq_order == 64:
                    output[:, seq_order - 1, :] = feed_forward_output[:, -1, :]
                else:
                    decoder_embedding[:, seq_order, :] = feed_forward_output[:, -1, :]
                    output[:, seq_order - 1, :] = feed_forward_output[:, -1, :]

        return output


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, in_dim,
                 out_dim, is_same_domain=True, input_vocab_size=None, output_vocab_size=None):
        super(Transformer, self).__init__()

        if is_same_domain:

            assert (input_vocab_size is None or output_vocab_size is None), \
                'setting is same domain but you have input_vocab or output_vocab'
            self.embedding = Embedding(vocab_size, in_dim)

        else:
            self.encoder_embedding = Embedding(vocab_size, in_dim)
            self.decoder_embedding = Embedding(vocab_size, in_dim)

        self.encoder_layers = nn.ModuleList(
            [Encoder(in_dim, out_dim, num_heads)
             for i in range(num_layers)]
        )

        self.decoder_layers = nn.ModuleList(
            [Decoder(in_dim, out_dim, num_heads)
             for i in range(num_layers)]
        )

        self.vocab_projection = nn.Linear(in_dim, vocab_size)
        self.is_same_domain = is_same_domain

    def forward(self, input_tokens, output_tokens=None):
        # embedding
        if self.is_same_domain:
            embedded_input = self.embedding(input_tokens)
            embedded_output = self.embedding(output_tokens)
        else:
            embedded_input = self.encoder_embedding(input_tokens)
            embedded_output = self.embedding(output_tokens)

        # encoding
        for i, encoder_layer in enumerate(self.encoder_layers):
            if i == 0:
                encoded_token = encoder_layer(embedded_input)
            else:
                encoded_token = encoder_layer(encoded_token)

        for i, decoder_layer in enumerate(self.decoder_layers):
            if i == 0:
                decoded_token = decoder_layer(encoded_token, embedded_output)
            else:
                decoded_token = decoder_layer(encoded_token, decoded_token)

        output = self.vocab_projection(decoded_token)
        return output
