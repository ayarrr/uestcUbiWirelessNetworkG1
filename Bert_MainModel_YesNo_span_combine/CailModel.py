from pytorch_transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
VERY_NEGATIVE_NUMBER = -1e29

def sequence_mask(sequence_length, max_length=None): # [batch_size, ]
    if max_length is None:
        max_length = sequence_length.data.max()
    #print ("sequence_length:",sequence_length.shape)
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand.float() < seq_length_expand.float())

def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var
    
class SelfAttentiveEncoder(nn.Module):

    def __init__(self, dropout = 0.3, hidden_dim = 300, att_dim = 350, att_hops = 4, dictionary = None):
        super(SelfAttentiveEncoder, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.attender = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim , att_dim, bias=False),
                nn.Tanh(),
                nn.Linear(att_dim, att_hops, bias=False),
                )
        self.att_hops = att_hops
        self.dictionary = dictionary
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.attender:
            if type(layer) == nn.Linear:
                init.kaiming_normal(layer.weight.data)
                if layer.bias is not None:
                    init.constant(layer.bias.data, val=0)

    def forward(self, inputs, length, words = None, display=False):
        # input is [bsz, len, 2*nhid]
        # length is [bsz, ]
        # words is [bsz, len]
        #print ("inputs:",inputs.shape)
        #print ("length:",length)
        bsz, l, nhid2 = inputs.size()
        #print ("nhid2:",nhid2)
        mask = sequence_mask(length, max_length=l) # [bsz, len]
        compressed_embeddings = inputs.view(-1, nhid2)  # [bsz*len, 2*nhid]
        
        #print ("compressed_embeddings:",compressed_embeddings.shape)
        alphas = self.attender(compressed_embeddings)  # [bsz*len, hop]
        alphas = alphas.view(bsz, l, -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, l))  # [bsz*hop, len]
        alphas = alphas.view(bsz, -1, l)  # [bsz, hop, len]

        mask = mask.unsqueeze(1).expand(-1, self.att_hops, -1) # [bsz, hop, len]
        alphas = alphas * mask.cuda().float() + 1e-20
        alphas = alphas / alphas.sum(2, keepdim=True) # renorm

        info = []
        if display:
            for i in range(bsz):
                s = '\n'
                for j in range(self.att_hops):
                    for k in range(unwrap_scalar_variable(length[i])):
                        s += '%s(%.2f) ' % (self.dictionary.itow(unwrap_scalar_variable(words[i][k])), unwrap_scalar_variable(alphas[i][j][k]))
                    s += '\n\n'
                info.append(s)
                
        #最终的注意力值
        supplements = {
                'attention': alphas, # [bsz, hop, len]
                'info': info,
                }
        
        #改写后的原始输出
        result = torch.bmm(alphas, inputs) # [bsz, hop, 2*nhid]

        return (result, supplements) 

class CailModel(BertPreTrainedModel):
    def __init__(self, config, answer_verification=False, hidden_dropout_prob=0.3):
        super(CailModel, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.qa_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.bilstm = torch.nn.GRU(input_size =config.hidden_size+2 , hidden_size = config.hidden_size, num_layers = 2, batch_first = False, dropout = 0, bidirectional = True)
        
        self.selfAttention = SelfAttentiveEncoder(hidden_dim = config.hidden_size+2, att_dim = config.hidden_size)
        
        self.attention = nn.Linear(config.hidden_size+2, 1)
        
        self.qa_outputs = nn.Linear(config.hidden_size*2, 2)
        self.apply(self._init_weights)#init_bert_weights
        self.answer_verification = answer_verification
        
        self.retionale_outputs = nn.Linear(config.hidden_size + 2, 1)
        self.doc_att = nn.Linear(config.hidden_size + 2, 1)
        self.yes_no_ouputs = nn.Linear(config.hidden_size + 2, 2)
        if self.answer_verification:
            
            self.unk_ouputs = nn.Linear(config.hidden_size, 1)
            
            
            self.ouputs_cls_3 = nn.Linear(config.hidden_size, 3)

            self.beta = 100
        else:
            # self.unk_yes_no_outputs_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.unk_outputs = nn.Linear(config.hidden_size, 1)
            
            self.yes_no_mlp = nn.Linear(config.hidden_size*2, 2)
    

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                unk_mask=None, yes_mask=None, no_mask=None, word_feature_list=None, named_entity_list=None):
        #第一层：Bert层
        #同时从Bert中读取序列输出，和一个池化输出
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        
        batch_size = sequence_output.size(0)
        seq_length = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        
        #第二层：融合层
        #并入字符级别的词性向量，和命名实体识别向量
        word_feature_list = word_feature_list.view(batch_size,seq_length,1)
        named_entity_list = named_entity_list.view(batch_size,seq_length,1)
        sequence_output = torch.cat([sequence_output, word_feature_list], -1)
        sequence_output = torch.cat([sequence_output, named_entity_list], -1)
        #print ("sequence_output2:",sequence_output.shape)  #形状为（8,384,770）
        
        #第三层：注意力层
        """
        length = np.array([seq_length]*batch_size)
        length = torch.from_numpy(length)
        sequence_output, _ = self.selfAttention(sequence_output, length)
        #print ("sequence_output2:",sequence_output.shape)  #形状为（8,384,770）
        """
        #sequence_output, attention_value = self.attention(sequence_output, seq_length, hidden_size)
        
        #第四层：双向LSTM层
        sequence_output2, _= self.bilstm(sequence_output)
        #print ("sequence_output:",sequence_output.shape)  #形状为（8, 384, 1536）
        
        #第四层：注意力层
        sequence_output_matrix = sequence_output.view(batch_size*seq_length, hidden_size + 2)
        rationale_logits = self.retionale_outputs(sequence_output_matrix)
        rationale_logits = F.softmax(rationale_logits)
        # [batch, seq_len]
        rationale_logits = rationale_logits.view(batch_size, seq_length)

        # [batch, seq, hidden] [batch, seq_len, 1] = [batch, seq, hidden]
        final_hidden = sequence_output*rationale_logits.unsqueeze(2)
        sequence_output = final_hidden.view(batch_size*seq_length, hidden_size + 2)
        
        #篇章级别的注意力层
        attention = self.doc_att(sequence_output)
        attention = attention.view(batch_size, seq_length)
        attention = attention*token_type_ids.float() + (1-token_type_ids.float())*VERY_NEGATIVE_NUMBER

        attention = F.softmax(attention, 1)
        attention_value = attention.unsqueeze(2)
        
        #print ("attention_value:", attention_value.shape)
        attention_output = attention_value*final_hidden
        #print ("attention_output:", attention_output.shape)
        
        #输出层1：MLP，输出片段概率
        logits = self.qa_outputs(sequence_output2)
        #print ("qa_logits:", logits.shape)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        #输出层2：MLP，输出Yes/No概率
        attention_pooled_output = attention_output.sum(1)
        yes_no_logits = self.yes_no_ouputs(attention_pooled_output)
        yes_logits, no_logits = yes_no_logits.split(1, dim=-1)
        
        """
        #sequence_output_matrix = sequence_output.view(batch_size*seq_length, hidden_size*2)
        sequence_output = torch.sum(sequence_output, 1)
        yes_no_logit = self.yes_no_mlp(sequence_output)
        yes_logits, no_logits= yes_no_logit.split(1, dim=-1)
        """
        
        #输出层2：MLP，输出无答案、YesOrNo概率
        unk_logits = self.unk_outputs(pooled_output)
    
        #对最终概率进行拼接：[0-511]为片段起始位置概率 [512]为Yes答案概率，[513]为No答案概率，[514]为拒答概率
        new_start_logits = torch.cat([start_logits, unk_logits, yes_logits, no_logits], 1)
        new_end_logits = torch.cat([end_logits, unk_logits, yes_logits, no_logits], 1)
        
        #如果仅为片段抽取类问题的训练，输出损失
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            #如果答案的维度大于1，则进行压缩
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            #将答案起止位置的取值，限制在（1,514）内
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(1, ignored_index)
            end_positions.clamp_(1, ignored_index)
            
            #使用交叉熵计算损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(new_start_logits, start_positions)
            end_loss = loss_fct(new_end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        #如果为评价预测，则直接输出概率
        else:
            #return start_logits, end_logits, unk_logits, yes_logits, no_logits
            return start_logits, end_logits, unk_logits, yes_logits, no_logits, attention_value


class MultiLinearLayer(nn.Module):
    def __init__(self, layers, hidden_size, output_size, activation=None):
        super(MultiLinearLayer, self).__init__()
        self.net = nn.Sequential()

        for i in range(layers-1):
            self.net.add_module(str(i)+'linear', nn.Linear(hidden_size, hidden_size))
            self.net.add_module(str(i)+'relu', nn.ReLU(inplace=True))

        self.net.add_module('linear', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)

