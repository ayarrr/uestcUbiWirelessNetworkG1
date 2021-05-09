# 项目背景
根据基金的历史涨跌数据，运用机器学习的方法，可以分析出基金之间的历史相关性。如果投资者再进行投资的话，可以参考这些相关性数据，从而避免投资高相关性的基金，获得更为稳健的投资。

# 方法介绍
由于数据的维度比较大，包含了61个时间步、每个时间步下有39个特征，因此使用常规机器学习模型需要使用特征提取。最开始的时候，我跟金融专业的同学交流了之后，尝试使用了技术因子，对基金的历史曲线进行特征提取。后来在看论文的时候，尝试使用了动态规整距离，直接比较两个基金的历史曲线的相似度。从结果上看，基于动态规整距离的特征提取方式更好。
后来，我尝试使用了深度学习模型。第一个深度模型是使用LSTM实现的Encoder-decoder模型，在编码器输出LSTM的隐层状态向量，然后输入到解码器中，同时将解码器的后面添加一个全连接层作为输出层，输出结果的维度为1，用于计算两个基金的相关性数值。这个Encoder-decoder模型的并不比lightgbm高出多少。
第二个深度学习模型融合例如注意力机制。在encoder-decoder的两个部分各加了一个注意力机制。encoder部分的注意力机制用于连接当前特征数值和LSTM的隐层状态向量，用于发掘长期依赖。Decoder部分的注意力机制用于对齐编码的特征数值和历史相似性数值，发掘特征和历史相似性数值之间的依赖。这种模型有了比较好的提升。


# 代码结构
## 机器学习方法
### 特征提取方法一： 只提取61个时间步的最后一个时间步
预处理文件：processing1.py
模型文件：lightgbm_processing1.py, MLP_processing1.py
预测文件：predict_processing1.py

### 特征提取方法二： 使用金融指标进行提取
预处理文件：processing2.py
模型文件：lightgbm_processing2.py, MLP_processing2.py
预测文件：predict_processing2.py

### 特征提取方法一： 使用DTW进行提取
预处理文件：processing3.py
模型文件：lightgbm_processing3.py, MLP_processing3.py
预测文件：predict_processing3.py

## 深度学习方法
### 关于LSTM的相关探索
预处理文件： all_data.py
单纯的LSTM： LSTM.py
以LSTM为基础构建的Encoder-decoder： LSTM_encoder_decoder.py
SAE(多层auto_Encoder), WT(小波降噪)： LSTM_SAE.py, LSTM_WT.py, LSTM_WT_encoder_decoder.py, LSTM_WT_SAE.py

### DA_RNN
预处理文件：all_data_da_RNN.py
单纯的DA_RNN： DA_RNN.py
改进的DA_RNN: DA_RNN2.py