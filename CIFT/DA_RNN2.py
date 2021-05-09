# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 22:58:50 2019

@author: 10097
"""

# -*- coding: utf-8 -*-

#from ops import *
from torch.autograd import Variable
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib 
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Encoder部分
'''
Encoder(
  (encoder_lstm): LSTM(128, 128)
  (encoder_attn): Linear(in_features=265, out_features=1, bias=True)
)
'''
class Encoder(nn.Module):
    #初始化Encoder 
    def __init__(self,T,input_size,encoder_num_hidden,parallel=False):
        super(Encoder,self).__init__()
        self.encoder_num_hidden=encoder_num_hidden
        self.input_size=input_size
        self.parallel=parallel
        self.T=T
        
        #图1，时序注意力机制：
        #Encoder是LSTM
        self.encoder_lstm=nn.LSTM(input_size=self.input_size,hidden_size=self.encoder_num_hidden)
        #构建输入注意力机制基于已知的注意力模型
        #公式8： W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn=nn.Linear(in_features=2*self.encoder_num_hidden+self.T-1,out_features=1,bias=True)
    
    #初始化所有0隐层状态和细胞状态为encoder
    def _init_states(self,X):
        #隐层状态和细胞状态[num_layers*num_directions,batch_size,hidden_size]
        print (X.size(0))
        initial_states=Variable(X.data.new(1,X.size(0),self.encoder_num_hidden).zero_())
        return initial_states
    
    def forward(self,X):
        X_tilde=Variable(X.data.new(X.size(0),self.T-1,self.input_size).zero_())
        X_encoded=Variable(X.data.new(X.size(0),self.T-1,self.encoder_num_hidden).zero_())
        #公式8，参数不在nn.Linear中，但需要学习
        #即v_e和U_e
        #隐层状态和细胞状态，使用维度hidden_size来初始化状态
        h_n=self._init_states(X)
        s_n=self._init_states(X)
        
        for t in range(self.T-1):
            #batch_size*input_size*(2*hidden_size+T-1) 
            #h_n.repeat(self.input_size,1,1) 重复self.input_size行
            x=torch.cat((h_n.repeat(self.input_size,1,1).permute(1,0,2),
                        s_n.repeat(self.input_size,1,1).permute(1,0,2),
                        X.permute(0,2,1)),dim=2)
            #print ("concat后的X:",x.shape)
            x=self.encoder_attn(x.view(-1,self.encoder_num_hidden*2+self.T-1))
            #通过softmax获得注意力权值
            alpha=F.softmax(x.view(-1,self.input_size))
            #print ("softmax后的alpha:",alpha.shape)
            #为LSTM重新做一个新输入
            x_tilde=torch.mul(alpha,X[:,t,:])
            #print ("x_tilde:",x_tilde.shape)
            
            #encoder LSTM
            self.encoder_lstm.flatten_parameters()
            _,final_state=self.encoder_lstm(x_tilde.unsqueeze(0),(h_n,s_n))
            h_n=final_state[0]
            s_n=final_state[1]
            #print ("lstm的新隐层状态h_n:",h_n.shape)
            
            X_tilde[:,t,:]=x_tilde
            X_encoded[:,t,:]=h_n
        return (X_tilde,X_encoded)
    
    

#decoder部分   
'''
Decoder(
  (attn_layer): Sequential(
    (0): Linear(in_features=384, out_features=128, bias=True)
    (1): Tanh()
    (2): Linear(in_features=128, out_features=1, bias=True)
  )
  (lstm_layer): LSTM(1, 128)
  (fc): Linear(in_features=129, out_features=1, bias=True)
  (fc_final): Linear(in_features=256, out_features=1, bias=True)
)
'''
class Decoder(nn.Module):
    #初始化decoder部分
    def __init__(self,T,decoder_num_hidden,encoder_num_hidden):
        super(Decoder,self).__init__()
        self.decoder_num_hidden=decoder_num_hidden
        self.encoder_num_hidden=encoder_num_hidden
        self.T=T
        #注意力层
        self.attn_layer=nn.Sequential(nn.Linear(2*decoder_num_hidden+encoder_num_hidden,encoder_num_hidden),nn.Tanh(),nn.Linear(encoder_num_hidden,1))
        self.lstm_layer=nn.LSTM(input_size=1,hidden_size=decoder_num_hidden)
        self.fc=nn.Linear(encoder_num_hidden+1,1)
        self.fc_final=nn.Linear(decoder_num_hidden+encoder_num_hidden,1)
        self.fc.weight.data.normal_()
        
    #初始化所有第0隐层状态和细胞状态
    def _init_states(self,X):
        initial_states=Variable(X.data.new(1,X.size(0),self.decoder_num_hidden).zero_())
        return initial_states
    
    def forward(self,X_encoded,y_prev):
        d_n=self._init_states(X_encoded)
        c_n=self._init_states(X_encoded)
        
        for t in range(self.T-1):
            x=torch.cat((d_n.repeat(self.T-1,1,1).permute(1,0,2),
                         c_n.repeat(self.T-1,1,1).permute(1,0,2),
                         X_encoded),dim=2)
            #print ("decoder_concat的形状为：",x.shape)
            beta=F.softmax(self.attn_layer(x.view(-1,2*self.decoder_num_hidden+self.encoder_num_hidden)).view(-1,self.T-1))
            #print ("注意力值beta的形状为：",beta.shape)
            
            #公式 14：计算上下文向量，batch_size*encoder_hidden_size
            #batch matrix multiply
            context=torch.bmm(beta.unsqueeze(1),X_encoded)[:,0,:]
            #print ("上下文向量context第一列的形状为：",context.shape)
            if t < self.T-1:
                #公式 15：batch_size*1
                y_tilde=self.fc(torch.cat((context,y_prev[:,t].unsqueeze(1)),dim=1))
                #print ("重写后的y_prev的形状为：",y_tilde.shape)
                #公式 16：LSTM
                self.lstm_layer.flatten_parameters()
                _,final_states=self.lstm_layer(y_tilde.unsqueeze(0),(d_n,c_n))
                #1*batch_size*decoder_num_hidden
                d_n=final_states[0]
                #1*batch_size*decoder_num_hidden
                c_n=final_states[1]
                #print ("编码后的y_prev的形状为：",d_n.shape)
                
        #公式 22：最终输出
        y_pred=self.fc_final(torch.cat((d_n[0],context),dim=1))
        #print ("最终输出的形状为:",y_pred.shape)
        return y_pred
    

class DA_rnn(nn.Module):
    def __init__(self,X,y_prev,y,T,encoder_num_hidden,decoder_num_hidden,batch_size,learning_rate,epochs,parallel=False):
        super(DA_rnn,self).__init__()
        self.encoder_num_hidden=encoder_num_hidden
        self.decoder_num_hidden=decoder_num_hidden
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.parallel=parallel
        self.shuffle=False
        self.epochs=epochs
        self.T=T
        self.X=X
        self.y=y
        self.y_prev=y_prev
        self.Encoder=Encoder(input_size=X.shape[-1],encoder_num_hidden=encoder_num_hidden,T=T)
        self.Decoder=Decoder(encoder_num_hidden=encoder_num_hidden,decoder_num_hidden=decoder_num_hidden,T=T)
        #损失函数
        self.criterion=nn.MSELoss()
        if self.parallel:
            self.encoder=nn.DataParallel(self.encoder)
            self.decoder=nn.DataParallel(self.decoder)
        self.encoder_optimizer=optim.Adam(params=filter(lambda p:p.requires_grad,self.Encoder.parameters()),lr=self.learning_rate)
        self.decoder_optimizer=optim.Adam(params=filter(lambda p:p.requires_grad,self.Decoder.parameters()),lr=self.learning_rate)
        #训练集
        #self.train_timesteps=int(self.X.shape[0]*0.7)
        self.train_timesteps=self.X.shape[0]
        self.input_size=self.X.shape[-1]
    
    def train_forward(self,X,y_prev,y_gt):
        #零梯度
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        input_weighted,input_encoded=self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
        y_pred=self.Decoder(input_encoded,Variable(torch.from_numpy(y_prev).type(torch.FloatTensor)))
        y_true=Variable(torch.from_numpy(y_gt).type(torch.FloatTensor))
        y_true=y_true.view(-1,1)
        loss=self.criterion(y_pred,y_true)
        loss.backward()
        
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return loss.item()
    
    def train(self):
        #encoder和decoder一起进行训练，都使用的是Adam优化器
        iter_per_epoch=int(np.ceil(self.train_timesteps*1.0/self.batch_size))
        self.iter_losses=np.zeros(self.epochs*iter_per_epoch)
        self.epoch_losses=np.zeros(self.epochs)
        n_iter=0
        loss=[]
        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx=np.random.permutation(self.train_timesteps)#-self.T
            else:
                ref_idx=np.array(range(self.train_timesteps))#-self.T
            idx=0
            while(idx < self.train_timesteps):
                #获取X_train的指数
                #indices=ref_idx[idx:(idx+self.batch_size)]
                if idx+self.batch_size < self.train_timesteps:
                    end=idx+self.batch_size
                else :
                    end=self.train_timesteps
                indices=ref_idx[idx:end]
                """
                x=np.zeros((len(indices),self.T-1,self.input_size))
                y_prev=np.zeros((len(indices),self.T-1))
                y_get=self.y[indices+self.T]
                
                #将X转化为三维的tensor 对X进行切片操作，切时间步为indices[bs]:(indices[bs]+9)的内容
                for bs in range(len(indices)):#总共且batch_size大小的内容
                    x[bs,:,:]=self.X[indices[bs]:(indices[bs]+self.T-1),:]
                    y_prev[bs,:]=self.y[indices[bs]:(indices[bs]+self.T-1)]
                """
                x=np.asarray(self.X[indices],dtype="float32")
                y_prev=np.asarray(self.y_prev[indices],dtype="float32")
                y_get=np.asarray(self.y[indices],dtype="float32")
                loss=self.train_forward(x,y_prev,y_get)
                self.iter_losses[int(epoch*iter_per_epoch+idx/self.batch_size)]=loss
                
                idx=end
                n_iter=n_iter+1
                
                if n_iter % 50000==0 and n_iter !=0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr']=param_group['lr']*0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr']=param_group['lr']*0.9
                self.epoch_losses[epoch]=np.mean(self.iter_losses[range(epoch*iter_per_epoch,(epoch+1)*iter_per_epoch)])
            if epoch % 1==0:#10
                print ("Epochs:",epoch,"Iterations:",n_iter," Loss:",self.epoch_losses[epoch])
                loss.append(self.epoch_losses[epoch])
            
        
            """
            if epoch==self.epochs-1:
                y_train_pred=self.test(on_train=True)
                y_test_pred=self.test(on_train=False)
                y_pred=np.concatenate((y_train_pred,y_test_pred))
                plt.ioff()
                plt.figure()
                plt.plot(range(1,1+len(self.y)),self.y,label="True")
                plt.plot(range(self.T,len(y_train_pred)+self.T),y_train_pred,label="Predicted-Train")
                plt.legend(loc="upper left")
                plt.show()
            """
        loss=np.asarray(loss)
        np.save("processing1/DA_RNN_train.npy",loss)
    
    def val(self):
        pass
    
    def test(self,X_test,y_prev_test,on_train=False):
        #预测的时候是encoder和decoder单独进行预测的
        if on_train:
            y_pred=np.zeros(self.train_timesteps-self.T+1)
        else:
            y_pred=np.zeros(X_test.shape[0])
        i=0
        while i<len(y_pred):
            if i+self.batch_size < len(y_pred):
                end=i+self.batch_size
            else:
                end=len(y_pred)
            batch_idx=np.array(range(len(y_pred)))[i:end]
            """
            X=np.zeros((len(batch_idx),self.T-1,self.X.shape[1]))
            y_history=np.zeros((len(batch_idx),self.T-1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j,:,:]=self.X[range(batch_idx[j],batch_idx[j]+self.T-1),:]
                    y_history[j,:]=self.y[range(batch_idx[j],batch_idx[j]+self.T-1)]
                else:
                    X[j,:,:]=self.X[range(batch_idx[j]+self.train_timesteps-self.T,batch_idx[j]+self.train_timesteps-1),:]
                    y_history[j,:]=self.y[range(batch_idx[j]+self.train_timesteps-self.T,batch_idx[j]+self.train_timesteps-1)]
            """
            X=np.asarray(X_test[batch_idx],dtype="float32")
            y_history=np.asarray(y_prev_test[batch_idx],dtype="float32")
            y_history=Variable(torch.from_numpy(y_history).type(torch.FloatTensor))
            _,input_encoded=self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)))
            y_pred[i:end]=self.Decoder(input_encoded,y_history).cpu().data.numpy()[:,0]
            i=end
        return y_pred

import all_data_da_RNN
from sklearn import metrics
def main():
    X,y_prev,y=all_data_da_RNN.get_trainset()
    
    #数据归一化
    X=(X-X.mean())/X.std()
    y=np.reshape(y,(y.shape[0],1))
    
    #参数设置
    batch_size_m=128
    hidden_encoder=128
    hidden_decoder=128
    T=62#原先需要61天，则T-1=61
    epochs_m=50
    lr=0.001
    
    #数据集划分
    #X_train,y_train,y_prev_train,X_val,y_val,y_prev_val=all_data_da_RNN.train_val_split(X,y_prev,y,T) 
    
    #初始化模型
    model=DA_rnn(X,y_prev,y,T,hidden_encoder,hidden_decoder,batch_size_m,lr,epochs_m)
    
    #训练
    model.train()
    
    #导入数据
    X_test,y_prev_test,y_test=all_data_da_RNN.get_testset()
    
    #进行数据归一化处理
    X_test=(X_test-X_test.mean())/X_test.std()
    y_test=np.reshape(y_test,(y_test.shape[0],1))
    
    #预测
    predict=model.test(X_test,y_prev_test)
    
    print ("MAE:",metrics.mean_absolute_error(y_test,predict))
    print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,predict)))
    print ("score:",GetScore(predict,y_test))
    
def GetScore(ypred,y_test):
    mae=0
    tmape=0
    n=len(y_test)
    for i in range(n):
        p=ypred[i]
        t=y_test[i]
        mae=mae+abs(p-t)
        tmape=tmape+abs((p-t)/(1.5-t))
    tmape=tmape/n
    return tmape
'''
    mae=mae/n
    score=(2/(2+mae+tmape))**2
'''

main()