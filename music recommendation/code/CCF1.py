# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:56:46 2019

@author: mliwang
CF大合集包含基于用户的CF,基于item的CF,不训练的svd,以及需要训练的MF
"""
import pandas as pd
from abc import ABCMeta, abstractmethod
import numpy as np
from collections import defaultdict
class CF_base(metaclass=ABCMeta):
    def __init__(self, k=3):
        self.k = k#控制推荐多少个结果
        self.n_user = None
        self.n_item = None
        self.user_ids=None#与列表对应
        self.item_ids=None
        self.user_matrix=None
        self.item_matrix=None
        self.predictUser=None
        self.x=None

    @abstractmethod
    def cal_prediction(self, *args):
        pass

    @abstractmethod
    def cal_recommendation(self, user_id, data):
        pass

    def fit(self,data,predictUser):
        # 计算所有用户的推荐物品
        self.init_param(data,predictUser)
        all_users = {}
        for i in self.predictUser:
            all_users[i]=self.cal_recommendation(i,self.x)
        return all_users#返回一个推荐列表
class CF_knearest(CF_base):
    """
    基于物品的K近邻协同过滤推荐算法
    """

    def __init__(self, k, criterion='cosine'):#data是一个dataframe，其中列为item,行为user
        super(CF_knearest, self).__init__(k)
        self.criterion = criterion
        self.simi_mat = None
        return
    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma 

    def init_param(self,data,predictUser):
         # 初始化参数
        self.n_user = data.shape[0]
        self.n_item = data.shape[1]
        self.predictUser=predictUser
        self.user_ids=list(data.index)
        self.item_ids=list(data.columns)
        x=data.values
        self.x=x
        x_normed =  self.standardization(x)
        self.simi_mat = self.cal_simi_mat(x_normed)
        return


    def cal_similarity(self, i, j, data):
        # 计算物品i和物品j的相似度
        items = data[:, [i, j]]
#        del_inds = np.where(items == 0)[0]
#        items = np.delete(items, del_inds, axis=0)
        if items.size == 0:
            similarity = 0
        else:
            v1 = items[:, 0]
            v2 = items[:, 1]
            if self.criterion == 'cosine':
                if np.std(v1) > 1e-3:  # 方差过大，表明用户间评价尺度差别大需要进行调整
                    v1 = v1 - v1.mean()
                if np.std(v2) > 1e-3:
                    v2 = v2 - v2.mean()
                similarity = (v1 @ v2) / np.linalg.norm(v1, 2) / np.linalg.norm(v2, 2)
            elif self.criterion == 'pearson':
                similarity = np.corrcoef(v1, v2)[0, 1]
            else:
                raise ValueError('the method is not supported now')
        return similarity

    def cal_simi_mat(self, data):
        # 计算物品间的相似度矩阵
        simi_mat = np.ones((self.n_item, self.n_item))
        for i in range(self.n_item):
            for j in range(i + 1, self.n_item):
                simi_mat[i, j] = self.cal_similarity(i, j, data)
                simi_mat[j, i] = simi_mat[i, j]
        return simi_mat

    def cal_prediction(self, user_row, item_ind):
        # 计算预推荐物品i对目标活跃用户u的吸引力
        purchase_item_inds = np.where(user_row >0)[0]
        rates = user_row[purchase_item_inds]#找到该用户曾经购买过的item
        simi = self.simi_mat[item_ind][purchase_item_inds]
        return np.sum(rates * simi) / np.linalg.norm(simi, 1)

    def cal_recommendation(self, user_ind, data):
        # 计算目标用户的最具吸引力的k个物品list
        item_prediction = defaultdict(float)
        user_ind=self.user_ids.index(user_ind)#找到对应行号
        user_row = data[user_ind]
        purchase_item_inds = np.where(user_row >0)[0]
        rates = user_row[purchase_item_inds]#找到该用户曾经购买过的item
        concer_simi=self.simi_mat[purchase_item_inds,:]
#        print(rates.shape,concer_simi.shape)
        item_prediction=rates.T@concer_simi#得到当前user对各个项目的可能点击率
        item_prediction=dict(zip(list(range(len(user_row))),(item_prediction)))  
        
#        un_purchase_item_inds = np.where(user_row == 0)[0]#找到0行，不算已经点击过的
#        for item_ind in range(len(user_row)):
#            item_prediction[item_ind] = self.cal_prediction(user_row, item_ind)#计算这个用户没买的商品和他的相似度
        res = sorted(item_prediction, key=item_prediction.get, reverse=True)
        result=res[:self.k]
        fr=[]
        for i in result:
            fr.append(self.item_ids[i])
            
        return fr
class CF_user(CF_base):
    """
    基于user的K近邻协同过滤推荐算法
    """

    def __init__(self, k, criterion='cosine'):#data是一个dataframe，其中列为item,行为user
        super(CF_user, self).__init__(k)
        self.criterion = criterion
        self.simi_mat = None
        return
    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma 

    def init_param(self,data,predictUser):
         # 初始化参数
        self.n_user = data.shape[0]
        self.n_item = data.shape[1]
        self.predictUser=predictUser
        self.user_ids=list(data.index)
        self.item_ids=list(data.columns)
        x=data.values
        self.x=x
        x_normed =  self.standardization(x)
        self.simi_mat = self.cal_usersimi_mat(x_normed)
        return


    def cal_similarity(self, i, j, data):
        # 计算物品i和物品j的相似度
        items = data[:, [i, j]]
        del_inds = np.where(items == 0)[0]
        items = np.delete(items, del_inds, axis=0)
        if items.size == 0:
            similarity = 0
        else:
            v1 = items[:, 0]
            v2 = items[:, 1]
            if self.criterion == 'cosine':
                if np.std(v1) > 1e-3:  # 方差过大，表明用户间评价尺度差别大需要进行调整
                    v1 = v1 - v1.mean()
                if np.std(v2) > 1e-3:
                    v2 = v2 - v2.mean()
                similarity = (v1 @ v2) / np.linalg.norm(v1, 2) / np.linalg.norm(v2, 2)
            elif self.criterion == 'pearson':
                similarity = np.corrcoef(v1, v2)[0, 1]
            else:
                raise ValueError('the method is not supported now')
        return similarity

    def cal_usersimi_mat(self, data):
        # 计算user间的相似度矩阵
        data=data.T
        simi_mat = np.ones((self.n_user, self.n_user))
        for i in range(self.n_user):
            for j in range(i + 1, self.n_user):
                simi_mat[i, j] = self.cal_similarity(i, j, data)
                simi_mat[j, i] = simi_mat[i, j]
        return simi_mat

    def cal_prediction(self, user_row, item_ind):
        # 计算预推荐物品i对目标活跃用户u的吸引力
        purchase_item_inds = np.where(user_row != 0)[0]
        rates = user_row[purchase_item_inds]
        simi = self.simi_mat[item_ind][purchase_item_inds]
        return np.sum(rates * simi) / np.linalg.norm(simi, 1)

    def cal_recommendation(self, user_ind, data):
        # 计算目标用户的最具吸引力的k个物品list
        user_ind=self.user_ids.index(user_ind)#找到对应行号
        user_row = data[user_ind]
        
        u_u_simi=self.simi_mat[user_ind]#1x n_user
            
        
#        un_purchase_item_inds = np.where(user_row == 0)[0]#找到0行,都有可能是将来要买的
#        alluI=[]
#        for item_ind in un_purchase_item_inds:
#            alluI.append(data[:,item_ind])
#        alluI=np.array(alluI)#  n_user X r_i
        
        alluI=data
#        print(u_u_simi.shape,alluI.shape)
        item_prediction=u_u_simi.T@alluI#得到各个未买项目的评分
        item_prediction=dict(zip(list(range(len(user_row))),(item_prediction)))    
        res = sorted(item_prediction, key=item_prediction.get, reverse=True)
        result=res[:self.k]
        fr=[]
        for i in result:
            fr.append(self.item_ids[i])
            
        return fr
class CF_svd(CF_base):
    """
    基于svd的协同过滤算法
    """

    def __init__(self, k=3, r=3):
        super(CF_svd, self).__init__(k)
        self.r = r  # 选取前k个奇异值
        self.uk = None  # 用户的隐因子向量
        self.vk = None  # 物品的隐因子向量
        return

    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma 

    def init_param(self,data,predictUser):
         # 初始化参数
        self.n_user = data.shape[0]
        self.n_item = data.shape[1]
        self.predictUser=predictUser
        self.user_ids=list(data.index)
        self.item_ids=list(data.columns)
        x=data.values
        self.x=x
        x_normed =  self.standardization(x)
        self.svd_simplify(x_normed)
        return
    
    def svd_simplify(self, data):
        # 奇异值分解以及简化
        u, s, v = np.linalg.svd(data)
        u, s, v = u[:, :self.r], s[:self.r], v[:self.r, :]  # 简化
        sk = np.diag(np.sqrt(s))  # r*r
        self.uk = u @ sk  # m*r
        self.vk = sk @ v  # r*n
        return

    def cal_prediction(self, user_ind, item_ind, user_row):
        rate_ave = np.mean(user_row)  # 用户已购物品的评价的平均值(未评价的评分为0)
        return rate_ave + self.uk[user_ind] @ self.vk[:, item_ind]  # 两个隐因子向量的内积加上平均值就是最终的预测分值

    def cal_recommendation(self, user_ind, data):
        # 计算目标用户的最具吸引力的k个物品list
        item_prediction = defaultdict(float)
        user_ind=self.user_ids.index(user_ind)#找到对应行号
        user_row = data[user_ind]
#        un_purchase_item_inds = np.where(user_row == 0)[0]
        for item_ind in range(len(user_row)):
            item_prediction[item_ind] = self.cal_prediction(user_ind, item_ind, user_row)
        res = sorted(item_prediction, key=item_prediction.get, reverse=True)
        result=res[:self.k]
        fr=[]
        for i in result:
            fr.append(self.item_ids[i])
            
        return fr
class M_F():
    """
    基于矩阵分解的推荐
    """
    def __init__(self,trainR,k,concenUser):
        self.K= k  # 推荐k个
        self.trainR=trainR
        self.R=trainR.loc[concenUser,:]#拿到需要关心的user的内容点击情况
        self.N,self.M=self.R.shape#user总数,item总数
        self.D=30#表示向量的维度
        
        self.uk = np.random.rand(self.N, self.D) #随机生成一个 N行 K列的矩阵  用户的隐因子向量
        self.vk = np.random.rand(self.D, self.M) #随机生成一个 M行 K列的矩阵  物品的隐因子向量
        return
    
    def matrix_fac(self,alpha, beta,steps=200):
        '''
        alpha 学习率
        beta 惩罚系数
        '''
        R=self.R.values#R为输入的评分矩阵(流行度矩阵)
#        result=[]
     
        for step in range(steps):
        #使用梯度下降的一步步的更新P,Q矩阵直至得到最终收敛值
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j]>0:
                        # .dot(P,Q) 表示矩阵内积,即Pik和Qkj k由1到k的和eij为真实值和预测值的之间的误差,
                        eij=R[i][j]-np.dot(self.uk[i,:],self.vk[:,j]) 
            
                        for k in range(self.D):
                            #在更新p,q时我们使用化简得到了最简公式
                            self.uk[i][k]=self.uk[i][k]+alpha*(2*eij*self.vk[k][j]-beta*self.uk[i][k])
                            self.vk[k][j]=self.vk[k][j]+alpha*(2*eij*self.uk[i][k]-beta*self.vk[k][j])
            
            loss = 0.0           
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j]>0: 
                        error = 0.0
                        for k in range(self.D):
                            error = error + self.uk[i,k]*self.vk[k,j]
                        loss = (R[i,j] - error) * (R[i,j] - error)
                        for k in range(self.D):
                            loss = loss + beta * (self.uk[i,k] * self.uk[i,k] + self.vk[k,j] * self.vk[k,j]) / 2
                            
            
            print('迭代轮次:', step, '   loss:', loss)
#            result.append(loss)#将每一轮更新的损失函数值添加到数组result末尾
            
            #当损失函数小于一定值时，迭代结束
            if loss<0.00001:
                break
        print("Train done!",step,loss)
        MF = np.dot(self.uk,self.vk)
        prdictR=pd.DataFrame(MF,index=self.R.index,columns=self.R.columns)
        userInterest=prdictR.apply(lambda x:self.getTopN(x,self.K),axis=1).T.to_dict(orient="list")
        return userInterest
    def getTopN(self,raw,n):#对于一个用户，已知他的一个TFIDF分布，获取topN的兴趣
        t=raw.nlargest(n).index.values
        d={}
        for i in range(n):
            d[str(i+1)]=t[i]
        return pd.Series(d)
class BPR:
    '''
    BPR(n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId')
    
    Bayesian Personalized Ranking Matrix Factorization (BPR-MF). During prediction time, the current state of the session is modelled as the average of the feature vectors of the items that have occurred in it so far.
        
    Parameters
    --------
    n_factor : int
        The number of features in a feature vector. (Default value: 100)
    n_iterations : int
        The number of epoch for training. (Default value: 10)
    learning_rate : float
        Learning rate. (Default value: 0.01)
    lambda_session : float
        Regularization for session features. (Default value: 0.0)
    lambda_item : float
        Regularization for item features. (Default value: 0.0)
    sigma : float
        The width of the initialization. (Default value: 0.05)
    init_normal : boolean
        Whether to use uniform or normal distribution based initialization.
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    
    '''
    def __init__(self, n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_item = 0.0, sigma = 0.05, init_normal = False, session_key = 'SessionId', item_key = 'ItemId'):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_item = lambda_item
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.item_key = item_key
        self.current_session = None

    def init(self, data):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_items, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_items, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_items)
    
    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_item * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_item * iF2)
        return np.log(sigm)
    
    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(self.n_items)}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.init(data)
        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.ItemIdx.values[e]
                iidx2 = data.ItemIdx.values[np.random.randint(self.n_items)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            print(it, np.mean(c))
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''      
        iidx = self.itemidmap[input_item_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)
        uF = self.I[self.session].mean(axis=0)
        iIdxs = self.itemidmap[predict_for_item_ids]
        return pd.Series(data=self.I[iIdxs].dot(uF) + self.bI[iIdxs], index=predict_for_item_ids)
             
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
def evaluate_sessions(pr, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):    
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    
    '''
    test_data.sort_values([session_key, time_key], inplace=True)#测试数据按user,时间排序
    items_to_predict = train_data[item_key].unique()#拿到所有的item
    evalutation_point_count = 0#测试次数
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0.0, 0.0
    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]#当前sessionId
        iid = test_data[item_key].values[i]#当前第i行的item
        if prev_sid != sid:#记录sessionId
            prev_sid = sid
        else:#同一个session
            if items is not None:
                if np.in1d(iid, items): items_to_predict = items
                else: items_to_predict = np.hstack(([iid], items))      
            preds = pr.predict_next(sid, prev_iid, items_to_predict)#给当前userid,prev_iid当前session中之前的item，items_to_predict为需要预测得分的item
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
            rank = (preds > preds[iid]).sum()+1
#            if iid in list(preds.index):
#                rank = (preds > preds[iid]).sum()+1
#            else:
#                print(iid,"*******",list(preds.index))
#                rank =cut_off+1
            assert rank > 0
            if rank < cut_off:
                recall += 1
                mrr += 1.0/rank
            evalutation_point_count += 1
        prev_iid = iid
    return recall/evalutation_point_count, mrr/evalutation_point_count