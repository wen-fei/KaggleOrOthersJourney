#-*- coding:utf-8 -*-

"""
大体思路：信号强的相关联
"""

import pandas as pd
from collections import defaultdict

user_shop_hehavior = pd.read_csv('G:\PythonProjects/tianchi\shop\ccf_first_round_user_shop_behavior.csv')
evalution = pd.read_csv('G:\PythonProjects/tianchi\shop\evaluation_public.csv')

#让WIFI关联商铺

#构造规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1

right_count = 0
for line in user_shop_hehavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in wifi_to_shops[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count/len(user_shop_hehavior)) #线下验证

#预测
preds = []
for line in evalution.values:
    index = 0
    while True:
        try:
            if index==5:
                pred_one = None
                break
            wifi = sorted([wifi.split('|') for wifi in line[6].split(';')],key=lambda x:int(x[1]),reverse=True)[index]
            counter = defaultdict(lambda : 0)
            for k,v in wifi_to_shops[wifi[0]].items():
                counter[k] += v
            pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
            break
        except:
            index+=1
    preds.append(pred_one)

result = pd.DataFrame({'row_id':evalution.row_id,'shop_id':preds})
result.fillna('s_666').to_csv('wifi_baseline.csv',index=None) #随便填的 这里还能提高不少
result.to_csv("submission_v1.csv", index=False)
