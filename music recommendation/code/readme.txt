1.calculateDetail.py  发现歌曲数量分布（output.png），并划分训练集与验证集
2.findtrick.py    获取所有需要考虑的歌曲的所有信息（歌单名，跟随者数量，歌曲详情。。。）
3.lda 训练歌单的主题分布 findbetternodel.py   doc_topic.csv记录了文档在各主题上的概率分布，
topic_word.csv记录了各个主题下最可能出现的前10个词，topic_word1.csv记录了各个主题中各个词的概率分布
4.classifyBytopic.py 给各个歌单定主题（选前n个）,最终的Pid_Topic.csv记录了各个主题下的pid

5.分主题讨论，看各个主题下合适的PW、AW、TW,并记录最好的 discussInDiffrentTopic.py

6.从1万个歌单中随机抽100个构成song-HIN考虑准确度DisscussINRandomTopic.py

7.baselines   CCF1   MYselfBaselines.py

