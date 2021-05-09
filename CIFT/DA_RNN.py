# -*- coding: utf-8 -*-
import numpy as np
from keras import layers
from keras import models
from keras.engine.topology import Layer
from sklearn import metrics

class Encoder(Layer):
    def __init__(self,t,input_size,hidden_encoder,**kwargs):
        self.t=t
        self.input_size=input_size
        self.hidden_encoder=hidden_encoder
        
        #构建输入注意力机制基于已知的注意力模型
        #公式8： W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn=layers.Dense(1,input_shape=(2*hidden_encoder+self.t-1,),use_bias=True)
        
        #encoder LSTM 
        self.encoder_lstm=layers.LSTM(self.hidden_encoder,return_state=True)
        super(Encoder,self).__init__(**kwargs)
    
    #https://stackoverflow.com/questions/46234722/initialize-keras-placeholder-as-input-to-a-custom-layer
    """
    def build(self,input_shape):
        assert len(input_shape)==3
        self.h_n=self.add_weight(name="h_n",shape=(self.t-1,self.hidden_encoder),
                                 initializer="zero",trainable=True)
        self.c_n=self.add_weight(name="c_n",shape=(self.t-1,self.hidden_encoder),
                                 initializer="zero",trainable=True)
        super(Encoder,self).build(input_shape)
    """  
    
    def init_state(self,X):
        init_state=layers.Input(tensor=X[:,0,:])
        init_state=layers.RepeatVector(self.hidden_encoder)(init_state)[:,:,0]
        init_state=layers.Reshape((init_state.shape[1],))(init_state)
        return (init_state)
    
    def call(self,X):
        """
        参数解释：
            X.shape[0]为训练批次
            X_tilde:保存最终结合注意力值重新表示的x_n向量 (X.shape[0],T-1,input_size)
            X_encoded:保存用LSTM对x_tilde的编码结果 (X.shape[0],T-1,input_size)
            h_n:encoder中LSTM的隐层状态 (X.shape[0],hidden_encoder)
            c_n:encoder中LSTM的细胞状态 (X.shape[0],hidden_encoder)
            encoder_concat:h_n,c_n,X的连接 (X.shape[0],input_size,2*hidden_encoder+T-1)
            alpha:encoder中的注意力值 (X.shape[0],input_size)
            x_tilde:结合注意力值重写的x_n向量 (X.shape[0],input_size)
        """
        
        #初始化新的X_n矩阵和X的编码矩阵
        X_tilde=[]
        X_encoded=[]
        
        #以形状为(X.shape[0],hidden_encoder)初始化隐层状态和细胞状态
        h_n=self.init_state(X)
        c_n=self.init_state(X)
        for t in range(self.t-1):
            #将输入整合到一起
            h_n_repeat=layers.RepeatVector(self.input_size)(h_n)
            #h_n=layers.Permute(1,0,2)(h_n)
            c_n_repeat=layers.RepeatVector(self.input_size)(c_n)
            #c_n=layers.Permute(1,0,2)(c_n)
            X_use=layers.Permute((2,1))(X)
            encoder_concat=layers.Concatenate(axis=2)([h_n_repeat,c_n_repeat,X_use])
            #print ("encoder_concat的形状为：",encoder_concat.shape)
            encoder_concat=layers.Reshape((-1,self.hidden_encoder*2+self.t-1))(encoder_concat)
            #通过softmax获得注意力权值
            alpha=self.encoder_attn(encoder_concat)
            alpha=layers.Reshape((-1,self.input_size))(alpha)
            alpha=layers.Activation("softmax")(alpha)
            #print ("alpha的形状为：",alpha.shape)
            
            #为LSTM重新做一个输入
            x_tilde=layers.Multiply()([alpha,X[:,t,:]])
            #print ("新的x_n向量的形状为：",x_tilde.shape)
            
            #encoder LSTM
            _,h_n,c_n=self.encoder_lstm(x_tilde,initial_state=(h_n,c_n))
            
            #将新的x_n向量和X的编码向量加入矩阵中
            x_tilde_repeat=layers.RepeatVector(1)(x_tilde[0])
            h_n_change=layers.RepeatVector(1)(h_n)
            X_tilde.append(x_tilde_repeat)
            X_encoded.append(h_n_change)
        
        X_tilde=layers.Concatenate(axis=1)(X_tilde)
        X_encoded=layers.Concatenate(axis=1)(X_encoded)
        #print ("X_tilde的形状为：",X_tilde.shape)
        #print ("X_encoded的形状为：",X_encoded.shape)
        return [X_tilde,X_encoded]
        
    def compute_output_shape(self,input_shape):
        return [(input_shape[0],self.t-1,self.input_size),(input_shape[0],self.t-1,self.hidden_encoder)]

class Decoder(Layer):
    def __init__(self,t,hidden_decoder,hidden_encoder,**kwargs):
        self.t=t
        self.hidden_decoder=hidden_decoder
        self.hidden_encoder=hidden_encoder
        #注意力层
        self.attn_layer=models.Sequential()
        self.attn_layer.add(layers.Dense(hidden_encoder,input_shape=(2*hidden_decoder+hidden_encoder,),activation="tanh"))
        self.attn_layer.add(layers.Dense(1,input_shape=(hidden_encoder,)))
        #decoder LSTM
        self.lstm_layer=layers.LSTM(hidden_decoder,input_dim=1,return_state=True)
        
        self.fc=layers.Dense(1,input_shape=(hidden_encoder+1,))
        self.fc_final=layers.Dense(1,input_shape=(hidden_encoder+hidden_decoder,))
        super(Decoder,self).__init__(**kwargs)
    
    def init_state(self,X):
        init_state=layers.Input(tensor=X[:,0,:])
        init_state=layers.RepeatVector(self.hidden_decoder)(init_state)[:,:,0]
        init_state=layers.Reshape((init_state.shape[1],))(init_state)
        return (init_state)
    
    def call(self,inputs):
        """
        参数解释：
            X.shape[0]为输入批次
            X_encoded:encoder传递过来的x的编码矩阵 (X.shape[0],T-1,hidden_encoder)
            y_prev:之前的输出序列 (X.shape[0],T-1)
            d_n:decoder中LSTM的隐层状态 (X.shape[0],hidden_decoder)
            c_n:decoder中LSTM的细胞状态 (X.shape[0],hidden_decoder)
            decoder_concat:d_n,c_n,X_encoded的连接 (X.shape[0],T-1,2*hidden_decoder+hidden_encoder)
            beta:decoder中的注意力值 (X.shape[0],hidden_encoder)
            context:通过结合beta与X_encoded得到的上下文向量 (X.shape[0],hidden_encoder)
            y_tilde:通过集合上下文向量与之前输出得到的重写y_t-1 (X.shape[0],1)
            d_n:使用LSTM编码y_tilde得到的之前输出的编码向量 (X.shape[0],hidden_decoder)
            y_prev:最终预测值 (X.shape[0],1)    
        """
        assert isinstance(inputs,list)
        X_encoded=inputs[0]
        y_prev=inputs[1]
        
        #以形状为(X.shape[0],hidden_decoder)初始化隐层状态和细胞状态
        d_n=self.init_state(X_encoded)
        c_n=self.init_state(X_encoded)
        
        for t in range(self.t-1):
            d_n_repeat=layers.RepeatVector(self.t-1)(d_n)
            #d_n=layers.Permute(1,0,2)(d_n)
            c_n_repeat=layers.RepeatVector(self.t-1)(c_n)
            #c_n=layers.Permute(1,0,2)(c_n)
            decoder_concat=layers.Concatenate(axis=-1)([d_n_repeat,c_n_repeat,X_encoded])
            #print ("decoder_concat的形状为：",decoder_concat.shape)
            decoder_concat=layers.Reshape((-1,2*self.hidden_decoder+self.hidden_encoder))(decoder_concat)
            #计算时序注意力值
            beta=self.attn_layer(decoder_concat)
            beta=layers.Reshape((-1,self.t-1))(beta)
            #print ("beta的形状为：",beta.shape)
            
            #公式 14：计算上下文向量，
            context=layers.Dot(axes=[2,1])([beta,X_encoded])
            context=context[:,0,:]
            #print ("context的形状为：",context.shape)
            if t<self.t-1:
                #公式15
                y_prev_repeat=layers.Reshape((1,))(y_prev[:,t])
                decoder_concat2=layers.Concatenate(axis=1)([context,y_prev_repeat])
                y_tilde=self.fc(decoder_concat2)
                #print ("y_tilde的形状为：",y_tilde.shape)
                #公式16
                y_prev_change=layers.RepeatVector(1)(y_tilde)
                _,d_n,c_n=self.lstm_layer(y_prev_change,initial_state=(d_n,c_n))
                #print ("y_n的编码d_n的形状为：",d_n.shape)
        """
            问题被抽象为：y_t=F(y_1,y_2,...,y_t-1,x_1,x_2,...,x_t)
            即通过之前输出序列和特征序列来预测当前输出
            由于历史数据过多，可以考虑设置一定窗口，但是窗口内包含与当前输出无关的历史数据，如上一周期的数据，因此可以使用注意力值对数据进行重写
            使用与输出相关的注意力机制对x_1,...,x_t进行编码，同时使用另一个与时间相关的注意力机制对y_1,...,y_t-1进行编码
            将两个编码结果进行链接，得到y_t=F(d_n,context)
            使用一个Dense对函数F进行拟合
            得到 y_t=fc_final(d_n,context)
        """
        
        #公式22，最终输出
        decoder_concat2=layers.Concatenate(axis=1)([d_n,context])
        y_pred=self.fc_final(decoder_concat2)
        #print ("最终输出y_pred的形状为：",y_pred.shape)
        return y_pred
    
    def compute_output_shape(self,input_shape):
        assert isinstance(input_shape,list)
        return (input_shape[0][0],1)

import all_data_da_RNN
from keras.optimizers import Adam
from keras import callbacks
def train():
    X,y_prev,y=all_data_da_RNN.get_trainset()
    
    #数据归一化
    X=(X-X.mean())/X.std()
    y=np.reshape(y,(y.shape[0],1))
    
    #参数设置
    input_size=X.shape[2]
    batch_size_m=30
    hidden_encoder=128
    hidden_decoder=128
    T=X.shape[1]+1
    epochs_m=50
    
    #数据集划分
    X_train,y_train,y_prev_train,X_val,y_val,y_prev_val=all_data_da_RNN.train_val_split(X,y_prev,y,T) 
    
    #DA_RNN
    X_input=layers.Input(shape=(X_train.shape[1],X_train.shape[2],))
    y_prev=layers.Input(shape=(T-1,))
    h_n,input_encoded=Encoder(T,input_size,hidden_encoder)(X_input)
    y_pred=Decoder(T,hidden_decoder,hidden_encoder)([input_encoded,y_prev])
    #print ("输入形状为：",X_input.shape," ",y_prev.shape)
    #print ("输出形状为：",y_pred.shape)
    
    model=models.Model(inputs=[X_input,y_prev],outputs=y_pred)
    
    #模型编译
    model.compile(optimizer=Adam(lr=0.001),loss="mae",metrics=['accuracy'])
    
    #自动调整学习率
    lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)
    
    #模型训练
    model.fit([X_train,y_prev_train],y_train,batch_size=batch_size_m,epochs=epochs_m,validation_data=([X_val,y_prev_val],y_val),callbacks=[lr,])
    
    #模型保存
    model.save("model/DA_RNN.h5")
    
    json_string = model.to_json()  #等价于 json_string = model.get_config()
    open('model/DA_RNN.json','w').write(json_string)  
    model.save_weights('model/DA_RNN_weight.h5') 
    
    #导入数据
    x_test,y_prev_test,y_test=all_data_da_RNN.get_testset()
    
    #进行数据归一化处理
    x_test=(x_test-x_test.mean())/x_test.std()
    y_test=np.reshape(y_test,(y_test.shape[0],1))
    
    #进行预测
    predict=model.predict([x_test,y_prev_test])
    
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

train()