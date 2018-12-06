import torch
from config import config
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

_config = config()


def count_labels(nlist):
    labels = 0
    index_list = []
    for alist in nlist:
        # clean the tags to make it uniform
        tags = []
        for i in range(0, len(alist)):
            if alist[i] != "O":
                tags.append(alist[i].split('-')[1])
            else:
                tags.append(alist[i])
        tags.append("O")

        # group the tags
        tag_group = []
        start = 0
        for i in range(1, len(tags)):
            if tags[i - 1] == tags[i]:
                start = i - 1
                continue
            stop = i - 1
            tag_group.append([tags[i - 1], start, stop])
            start = i

        # remove unwanted tags "O"
        tag_group[:] = (value for value in tag_group if value[0] != "O")
        #print('tag_group',tag_group)
        labels += len(tag_group)
        index_list.append(tag_group)

    return index_list, labels


def count_match(golden, predict):
    match_count = 0
    for i in range(0, len(golden)):
        for j in range(0, len(predict[i])):
            if predict[i][j] in golden[i]:
                #print('predict',golden[i], predict[i][j])
                match_count += 1

    return match_count


def evaluate(golden_list, predict_list):
    B = 1
    golden_labels, golden_count = count_labels(golden_list)
    #print('golden', golden_labels)
    predict_labels, predict_count = count_labels(predict_list)
    #print('pred', predict_labels)

    # TP
    match_count = count_match(golden_labels, predict_labels)

    #print(B)
    #print(precision)
    #print(recall)
    if match_count:
        # TP / (TP + FP)
        precision = match_count / predict_count
        # TP / (TP + FN)
        recall = match_count / golden_count
        F1 = ((B * B + 1) * precision * recall) / ((B * B * precision) + recall)
    else:
        if predict_count + golden_count == 0:
            F1 = 1.
        else:
            F1 = 0.
    #print('f1',F1)
    return F1


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    #ingate = F.sigmoid(ingate)
    #print(1-forgetgate)
    forgetgate = F.sigmoid(forgetgate)
    ingate = 1 - forgetgate
    #print(ingate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy
    #pass;

# word embedding dimension: [batch_size, max_sent_len, word_embedding_dim]
# max_sent_len: maximum length among sentences in the batch
# batch_char_index_matrices is a tensor that can be viewed as a list of matrices storing char_ids, where each matrix corresponds to a sentence,
# each sentence corresponds to a list of words, and each word corresponds to a list of char_ids.
# 1st dim: matrices, 2nd dim: word list of a sentence, 3rd dim: char ids
# batch_word_len_lists is tensor that can be viewed as a list of lists. Where each list corresponds to a sentence,
# and stores the length of each word.
# 1st dim: sentence list 2nd dim: length of each word
def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    # Given an input of the size [2,7,14], we will convert it a minibatch of the shape [14,14] to
    # represent 14 words(7 in each sentence), and 14 characters in each word.
    ## NOTE: Please DO NOT USE for Loops to iterate over the mini-batch.

    batch_size = len(batch_word_len_lists)
    max_word_len = max([max(i) for i in batch_word_len_lists])
    max_sent_len = max([len(i) for i in batch_word_len_lists])

    # 1. Reshape
    _14_14_resize = batch_char_index_matrices.resize(batch_size * max_sent_len, max_word_len)
    _14_resize = batch_word_len_lists.resize(batch_size * max_sent_len)
    #print(_14_14_resize.shape)

    # 2. Get corresponding char_Embeddings, we will have a Final Tensor of the shape [14, 14, 50]
    input_char_embeds = model.char_embeds(_14_14_resize)
    input_embeds = input_char_embeds
    #print(input_embeds.shape)

    # 3.Sort the the mini-batch wrt word-lengths, to form a pack_padded sequence.
    perm_idx, sorted_batch_word_len_list = model.sort_input(_14_resize)
    sorted_input_embeds = input_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)

    output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list.data.tolist(),
                                           batch_first=True)

    # 4.Feed the the pack_padded sequence to the char_LSTM layer.
    output_sequence, state = model.char_lstm(output_sequence)

    # 5. Get hidden state of the shape [2,14,50].
    output_sequence = pad_packed_sequence(output_sequence, batch_first=True)
    hidden_state = state[0]  # hidden, cell= state
    # 6. Recover the hidden_states corresponding to the sorted index.
    result = hidden_state[:, desorted_indices]

    # 7. Re-shape it to get a Tensor the shape [2,7,100].
    allcat = []

    for batch in range(0, batch_size):
        batch_concat = []
        begin = batch * max_sent_len
        end = max_sent_len * (batch + 1)
        #print(begin, end)
        for word_len in range(begin, end):
            direction_cat = torch.cat((result[0][word_len], result[1][word_len]), 0)
            batch_concat.append(direction_cat.detach().numpy())
        allcat.append(batch_concat)

    allcat = torch.Tensor(allcat)

    return allcat



