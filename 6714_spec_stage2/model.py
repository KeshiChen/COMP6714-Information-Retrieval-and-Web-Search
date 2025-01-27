# COMP6714 Project
# DO NOT MODIFY THIS FILE!!!
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from todo import new_LSTMCell, get_char_sequence


class sequence_labeling(nn.Module):
    def __init__(self, config, pretrain_word_embeddings, pretrain_char_embedding):
        super(sequence_labeling, self).__init__()

        self.config = config

        # employ the modified LSTM cell if the flag is True
        if self.config.use_modified_LSTMCell:
            torch.nn._functions.rnn.LSTMCell = new_LSTMCell

        self.word_embeds = nn.Embedding(self.config.nwords, self.config.word_embedding_dim)
        self.word_embeds.weight = nn.Parameter(torch.from_numpy(pretrain_word_embeddings).float())

        # below variants may be used for char embedding
        self.char_embeds = nn.Embedding(self.config.nchars, self.config.char_embedding_dim)
        self.char_embeds.weight = nn.Parameter(torch.from_numpy(pretrain_char_embedding).float())
        char_lstm_input_dim = self.config.char_embedding_dim
        self.char_lstm = nn.LSTM(char_lstm_input_dim, self.config.char_lstm_output_dim, 1, bidirectional=True)

        # employ char embedding if the flag is True
        if self.config.use_char_embedding:
            lstm_input_dim = self.config.word_embedding_dim + self.config.char_lstm_output_dim * 2
        else:
            lstm_input_dim = self.config.word_embedding_dim
        self.lstm = nn.LSTM(lstm_input_dim, self.config.hidden_dim, 1, bidirectional=True)

        self.lstm2tag = nn.Linear(self.config.hidden_dim * 2, self.config.ntags)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.non_recurrent_dropout = nn.Dropout(self.config.dropout)

    def sort_input(self, seq_len):
        # print("len",seq_len)
        seq_lengths, perm_idx = seq_len.sort(0, descending=True)
        return perm_idx, seq_lengths

    def _rnn(self, batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists):
        # [batch_size, words number, char number]
        # print("batch char", batch_char_index_matrices.shape)
        # [batch_size, words number]
        # print("batch word", batch_word_index_lists.shape)
        # print(batch_word_len_lists)
        input_word_embeds = self.word_embeds(batch_word_index_lists)

        # employ char embedding if the flag is True
        if self.config.use_char_embedding:
            #print("input_w_emb", input_word_embeds.shape)
            output_char_sequence = get_char_sequence(self, batch_char_index_matrices, batch_word_len_lists)
            #print(input_word_embeds.size(), output_char_sequence.size())
            input_embeds = self.non_recurrent_dropout(torch.cat([input_word_embeds, output_char_sequence], dim=-1))
        else:
            input_embeds = self.non_recurrent_dropout(input_word_embeds)
        # print("input_w_emb", input_word_embeds.shape)
        # print("input_emb",input_embeds.shape)
        perm_idx, sorted_batch_sentence_len_list = self.sort_input(batch_sentence_len_list)
        # print("perm_id", perm_idx)
        # print("bsl", batch_sentence_len_list)
        # print("sbsl", sorted_batch_sentence_len_list)
        # print("sorted_batch_sentence_len_list",sorted_batch_sentence_len_list)
        sorted_input_embeds = input_embeds[perm_idx]
        _, desorted_indices = torch.sort(perm_idx, descending=False)
        # print("des", desorted_indices)
        # print("shapes", sorted_input_embeds.size(), sorted_batch_sentence_len_list.size())
        output_sequence = pack_padded_sequence(sorted_input_embeds,
                                               lengths=sorted_batch_sentence_len_list.data.tolist(), batch_first=True)
        # print("out0", output_sequence.data.shape)
        # print("out", output_sequence)
        output_sequence, state = self.lstm(output_sequence)
        output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
        # print("out2", output_sequence.data.shape)
        output_sequence = output_sequence[desorted_indices]
        # print("out3", output_sequence.data.shape)
        output_sequence = self.non_recurrent_dropout(output_sequence)
        # print("out4", output_sequence.data.shape)
        # shape: [batch_size, words num, 100]
        logits = self.lstm2tag(output_sequence)
        return logits

    def forward(self, batch_word_index_lists, batch_sentence_len_list, batch_word_mask, batch_char_index_matrices,
                batch_word_len_lists, batch_char_mask, batch_tag_index_list):
        logits = self._rnn(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices,
                           batch_word_len_lists)
        batch_tag_index_list = batch_tag_index_list.view(-1)
        batch_word_mask = batch_word_mask.view(-1)
        logits = logits.view(-1, self.config.ntags)
        # print("log", logits.shape)
        train_loss = self.loss_func(logits, batch_tag_index_list) * batch_word_mask
        return train_loss.mean()

    def decode(self, batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists,
               batch_char_mask):
        logits = self._rnn(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices,
                           batch_word_len_lists)
        _, pred = torch.max(logits, dim=2)
        return pred
