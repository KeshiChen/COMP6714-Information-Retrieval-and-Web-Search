import torch
from config import config
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    match_count = count_match(golden_labels, predict_labels)

    precision = match_count / predict_count
    recall = match_count / golden_count
    #print(B)
    #print(precision)
    #print(recall)
    if precision and recall:
        F1 = ((B * B + 1) * precision * recall) / ((B * B * precision) + recall)
    else:
        F1 = 0.
    print('f1',F1)
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
    #batch_char_index_matrices = batch_char_index_matrices.view([7*config.batch_size, -1])
    #print("sizes",(batch_word_len_lists.size(), batch_char_index_matrices.size()))
    input_char_embeds = model.char_embeds(batch_char_index_matrices)
    input_embeds = input_char_embeds
    # 先排序每行，记录原始顺序,desort时逐行还原
    perm_idx_list = []
    sorted_batch_word_len_list = batch_word_len_lists.clone()
    for i, row in enumerate(batch_word_len_lists):
        pid, sorted_batch_word_len_list[i] = model.sort_input(batch_word_len_lists[i])
        perm_idx_list.append(pid.data.tolist())
    perm_idx_list = torch.LongTensor(perm_idx_list)
    #perm_idx, sorted_batch_word_len_list = model.sort_input(batch_word_len_lists)
    # print("bwl", batch_word_len_lists)
    # print("sbwl", sorted_batch_word_len_list)
    # print("pids", perm_idx_list)
    # 逐行获取sorted_input_embeds
    sorted_input_embeds = input_embeds.clone()
    for i, row in enumerate(input_embeds):
        sorted_input_embeds[i] = input_embeds[i][perm_idx_list[i]]
    # sorted_input_embeds = input_embeds[perm_idx]
    # 逐行获取desorted_indeces
    desorted_indices = perm_idx_list.clone()
    for i, row in enumerate(perm_idx_list):
        _, desorted_indices[i] = torch.sort(torch.LongTensor(perm_idx_list[i]), descending=False)
    #_, desorted_indices = torch.sort(perm_idx, descending=False)
    # 逐行获取output
    results = []
    for i, row in enumerate(sorted_input_embeds):
        #print("sie i",sorted_input_embeds[i].size())
        #print(sorted_batch_word_len_list[i].size())
        #print("sie", sorted_input_embeds[i][:,:,0])
        #print("sbwli", sorted_batch_word_len_list[i])
        output_sequence = pack_padded_sequence(sorted_input_embeds[i], lengths=[sorted_input_embeds.shape[2] for i in range(sorted_input_embeds.shape[1])], batch_first=True)
        output_sequence, state = model.char_lstm(output_sequence)
        output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
        output_sequence = output_sequence[desorted_indices[i]]
        #print("outseq", output_sequence.size())
        results.append(output_sequence.data.tolist())
    results = torch.Tensor(results)
    # print(sorted_input_embeds[:,:,0,:].view([config.batch_size,7,50]).size()) # 10,7,11,50 -> 10, 7, 50
    # print(sorted_batch_word_len_list)
    # #sorted_input_embeds = sorted_input_embeds[:,:,0,:].view([config.batch_size,7,config.char_embedding_dim])
    # output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list, batch_first=True)
    # print(output_sequence.data.size())
    # output_sequence, state = model.char_lstm(output_sequence)
    # output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
    # output_sequence = output_sequence[desorted_indices]
    # print("out", results.size())
    results = results[:,:,-1,:].view([batch_char_index_matrices.shape[0],-1,model.config.char_lstm_output_dim*2])
    #output_sequence = output_sequence.view([config.batch_size, 7, pad_length, config.char_lstm_output_dim*2])
    return results
def get_char_seaquence(model, batch_char_index_matrices, batch_word_len_lists):
    #todo: put each output into outputs tensor
    outputs = []
    print("aaa", batch_char_index_matrices.shape)
    for idx, sequence in enumerate(batch_word_len_lists):
        #print("word", sequence.shape)
        # input: sequence: [word1, word2, ....]
        # output: sorted sequence idx list by word length
        # word: character sequence
        input_char_embeds = model.char_embeds(batch_char_index_matrices[idx])
        #input_embeds= model.non_recurrent_dropout(input_char_embeds)
        input_embeds = input_char_embeds
        perm_idx, sorted_batch_word_len_list = model.sort_input(sequence)
        sorted_input_embeds = input_embeds[perm_idx]
        #print("sorted_input_embeds", sorted_input_embeds.shape)
        _, desorted_indices = torch.sort(perm_idx, descending=False)
        # pack_padded_sequence: https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099
        # https://blog.csdn.net/u012436149/article/details/79749409?utm_source=blogxgwz0
        #print("sbw",sorted_batch_word_len_list.data.tolist())
        output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
        #print("out:", output_sequence.data.shape)
        output_sequence, state = model.char_lstm(output_sequence)
        output_sequence, _ = pad_packed_sequence(output_sequence, batch_first=True)
        #print("out2:", output_sequence.data.shape)
        output_sequence = output_sequence[desorted_indices]
        #output_sequence = model.non_recurrent_dropout(output_sequence)
        #print(output_sequence.shape)
        outputs.append(output_sequence)
    for row in outputs:
        print(row.shape)
    outputs = tuple(outputs)
    print(len(outputs))
    result = torch.stack(outputs)
    print(result.shape)
    return result


