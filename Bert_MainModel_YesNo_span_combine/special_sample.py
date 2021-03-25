# -*- coding: utf-8 -*-
import json
def in_this_class(text, class_list):
    for one_label in class_list:
        if text.find(one_label) >= 0:
            return True
    return False

def class_filter(text):
    """
    key_question={"时间":["何时","那一天","哪一天","出生日期","出生年月","年龄","日期","何年","期限","什么时候","几月几日","几月","几号","几点","哪一年","哪年","哪个月","哪天","时间","星期几","多久","多长时间"],#When，how soon，how long
                  "谁":["什么人","哪些人","谁","何人","姓名"],#who
                  "地点":["什么地方","在哪","哪里","从哪","到哪","来哪","去哪","地方","何处","人在何处","地点","地址","何地","住址","位置","所在位置","单位"],#where
                  "实体":["哪个","哪","哪家","那家","哪些","那些","关系","物品","名称","同伙","何种","单位名称","车牌号","职位","职务","职业","职责","干什么","身份","类型","种类","来源","去向","下落","何物","车型","成分"],#which                      
                  "原因":["为什么","原因","怎么回事","为何","何事","为了","有何","因何","因","目的","为啥","缘由"],#why
                  "什么":["什么","有哪些","区别","排名","情况"],#what  
                  "方法":["如何","怎样","怎么","方法","攻略","以何","工作","内容","用何","手段","什么样","何用","何用途","方式","过程","经过"],#how
                  "数量":["几","多远","多久一次","多宽", "多少钱","多少","价格","票价","费用","几个","多长","几次","金额","数额","价值","多大","损伤","程度","损失","经费","几名","几人","含量","几年","几年级","几份"],#how many,how much，how often，how wide
                  "是非":["是否","是否是","有无"]
                  }
    """
    key_question=[["何时","那一天","哪一天","出生日期","出生年月","年龄","日期","何年","期限","什么时候","几月几日","几月","几号","几点","哪一年","哪年","哪个月","哪天","时间","星期几","多久","多长时间"],#When，how soon，how long
                  ["什么人","哪些人","谁","何人","姓名"],#who
                  ["什么地方","在哪","哪里","从哪","到哪","来哪","去哪","地方","何处","人在何处","地点","地址","何地","住址","位置","所在位置","单位"],#where
                  ["哪个","哪","哪家","那家","哪些","那些","关系","物品","名称","同伙","何种","单位名称","车牌号","职位","职务","职业","职责","干什么","身份","类型","种类","来源","去向","下落","何物","车型","成分"],#which                      
                  ["为什么","原因","怎么回事","为何","何事","为了","有何","因何","因","目的","为啥","缘由"],#why
                  ["什么","有哪些","区别","排名","情况"],#what 
                  ["如何","怎样","怎么","方法","攻略","以何","工作","内容","用何","手段","什么样","何用","何用途","方式","过程","经过"],#how
                  ["几","多远","多久一次","多宽", "多少钱","多少","价格","票价","费用","几个","多长","几次","金额","数额","价值","多大","损伤","程度","损失","经费","几名","几人","含量","几年","几年级","几份"],#how many,how much，how often，how wide
                  ["是否","是否是","有无"]
                  ]
    for i in range(len(key_question)):
        if in_this_class(text, key_question[i]):
            #该类别已满, 则不允许加入
            return (i)
    #没有对应类别，输出问题，不允许加入
    print ("没有对应的问题类别：",text)
    return (-1)

def special_sample_filter():
    with open("data/big_train_data.json", "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    input_data_len = len(input_data)
    #input_data_len = 10
    label_amount = 9
    
    #11个类别，每个类别提取100个样本
    #处理单个样本
    for label in range(label_amount):
        
        with open("data/big_train_data.json", "r", encoding='utf-8') as reader:
            one_data = json.load(reader)["data"]
        counter = 100
        for entry_index in range(input_data_len):
            entry = input_data[entry_index]
            #处理单个样本中的篇章
            delete_list1 = []
            print ("len(entry['paragraphs']):", len(entry["paragraphs"]))
            for paragraph_id in range(len(entry["paragraphs"])):
                paragraph = entry["paragraphs"][paragraph_id]
    
                #对问题进行字级别分词
                delete_list2 = []
                for qa_id in range(len(paragraph["qas"])):
                    question_text = paragraph["qas"][qa_id]["question"]
                    if class_filter(question_text) != label or counter == 0:
                        delete_list2.append(qa_id)
                        #del one_data[entry_index]["paragraphs"][paragraph_id]["qas"][qa_id]
                    else:
                        counter = counter -1
                
                print ("delete_list2:",delete_list2)
                for delete_index in range(len(delete_list2)):
                    one_delete = delete_list2[delete_index] - delete_index
                    del one_data[entry_index]["paragraphs"][paragraph_id]["qas"][one_delete]
                
                if len(one_data[entry_index]["paragraphs"][paragraph_id]["qas"]) == 0:
                    delete_list1.append(paragraph_id)
                    #del one_data[entry_index]["paragraphs"][paragraph_id]
            print ("delete_list1:",delete_list1)
            for delete_index in range(len(delete_list1)):
                one_delete = delete_list1[delete_index] - delete_index
                del one_data[entry_index]["paragraphs"][one_delete]
                
        #print ("one_data:",one_data)
        result_data = {}
        result_data['data'] = one_data
        result_data['version'] = "1.0"
        one_file = open("data/class_dev/class_dev"+str(label)+".json","w",encoding="utf8")
        json.dump(result_data, one_file)
        one_file.close()

def len_counter():
    label_amount = 9
    
    #11个类别，每个类别提取100个样本
    #处理单个样本
    for label in range(label_amount):
        with open("data/class_dev/class_dev"+str(label)+".json", "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
    
        input_data_len = len(input_data)
        #input_data_len = 10
        all_question = 0
        
        for entry_index in range(input_data_len):
            entry = input_data[entry_index]
            #处理单个样本中的篇章
            
            for paragraph_id in range(len(entry["paragraphs"])):
                paragraph = entry["paragraphs"][paragraph_id]
                
                all_question = all_question + len(paragraph["qas"])
        
        print ("类别"+str(label)+"的问题个数为：",all_question)
                
        
if __name__ == "__main__":
    #special_sample_filter()
    len_counter()

        


    