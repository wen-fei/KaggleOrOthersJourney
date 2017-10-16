# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import sys
import gc; gc.enable()
import collections
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/sample_submission_zero.csv')

transactions_features = ['msno', 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew', '_', '_', 'is_cancel']
trans_features = [feature for feature in transactions_features[1:] if feature != '_']
def make_transactions_features():
    print('loading...')
    infos = {}
    with open('../input/transactions.csv') as fd:
        count = 0
        fd.readline()
        for line in fd:
            pos = line.find(',')
            msid = line[:pos]
            splits = line[pos + 1:-1].split(',')
            info = [int(value) for value in splits[:]]
            info = [value for index, value in enumerate(info) if transactions_features[index + 1] != '_']
            if msid not in infos:
                infos[msid] = [[value] for value in info]
                infos[msid].insert(0, 1)
            else:
                infos[msid][0] += 1
                for index in range(1, 7):
                    infos[msid][index].append(info[index - 1])
            count += 1
            if count % 100000 == 0:
                print('processed: %d'%count)
    print('done: %d'%count)
    
    df_transactions = pd.DataFrame()
    df_transactions['msno'] = infos.keys()
    df_transactions['trans_count'] = [infos[key][0] for key in infos.keys()]
    for index, feature in enumerate(trans_features):
        df_transactions[feature] = [collections.Counter(infos[key][index + 1]).most_common()[0][0] for key in infos.keys()]
    
    return df_transactions

userlog_features = ['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs']
def make_userlog_features():
    print('loading...')
    infos = {}
    with open('../input/user_logs.csv') as fd:
        count = 0
        fd.readline()
        for line in fd:
            pos = line.find(',')
            msid = line[:pos]
            #_, num_25, num_50, num_75, num_985, num_100, num_unq, total_secs = [int(float(value)) for value in line[pos + 1:-1].split(',')]
            splits = line[pos + 1:-1].split(',')
            info = [int(value) for value in splits[:-1]]
            info.append(int(float(splits[-1])))
            #if len(info) != 8:
            #    print('not expect line: %s'%line[:-1])
            #    continue
            if msid not in infos:
                info[0] = 1
                infos[msid] = info
            else:
                infos[msid][0] += 1
                for index in range(1, 8):
                    infos[msid][index] += info[index]
            count += 1
            if count % 100000 == 0:
                print('processed: %d'%count)
    print('done: %d'%count)
    
    df_userlog = pd.DataFrame()
    df_userlog['msno'] = infos.keys()
    df_userlog['date_count'] = [infos[key][0] for key in infos.keys()]
    for index, feature in enumerate(userlog_features[1:]):
        if feature == 'total_secs':
            df_userlog[feature] = [infos[key][index]/3600 for key in infos.keys()]
        else:
            df_userlog[feature] = [infos[key][index] for key in infos.keys()]

    return df_userlog

transactions = make_transactions_features()
user_logs = make_userlog_features()

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')

train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')

members = pd.read_csv('../input/members.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')

gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

cols = [c for c in train.columns if c not in ['is_churn','msno']]

train.head()

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', sklearn.metrics.log_loss(labels, preds)

fold = 1
for i in range(fold):
    params = {
        'eta': 0.02, #use 0.002
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = sklearn.model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold
test['is_churn'] = pred.clip(0.0000001, 0.999999)
test[['msno','is_churn']].to_csv('submission3.csv', index=False)

