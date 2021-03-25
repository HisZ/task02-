

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
import lightgbm as lgb
import time 
from sklearn.preprocessing import OneHotEncoder
```


```python
train = pd.read_csv("./train.csv")
test = pd.read_csv("./testA.csv")
submit = pd.read_csv("./sample_submit.csv")
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 3 columns):
    id                   100000 non-null int64
    heartbeat_signals    100000 non-null object
    label                100000 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 2.3+ MB
    


```python
train.shape[0]
```




    100000




```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20000 entries, 0 to 19999
    Data columns (total 2 columns):
    id                   20000 non-null int64
    heartbeat_signals    20000 non-null object
    dtypes: int64(1), object(1)
    memory usage: 312.6+ KB
    


```python
y_train = train["label"]
data = pd.concat([train,test],axis=0)
x = data.drop(["id","label"],axis=1)
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=True'.
    
    To retain the current behavior and silence the warning, pass sort=False
    
      
    


```python
x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 120000 entries, 0 to 19999
    Data columns (total 1 columns):
    heartbeat_signals    120000 non-null object
    dtypes: object(1)
    memory usage: 1.8+ MB
    


```python
x = x["heartbeat_signals"].str.split(",",expand=True)
for col in x.columns:
    x[col] = x[col].astype("float64")
```


```python
x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 120000 entries, 0 to 19999
    Columns: 205 entries, 0 to 204
    dtypes: float64(205)
    memory usage: 188.6 MB
    


```python
x_train = x.head(train.shape[0])
x_test = x.tail(test.shape[0])
print(x_train.shape,x_test.shape)
```

    (100000, 205) (20000, 205)
    


```python
y_train.value_counts()
```




    0.0    64327
    3.0    17912
    2.0    14199
    1.0     3562
    Name: label, dtype: int64




```python
def abs_sum(y_pre,y_tru):
    y_pre=np.array(y_pre)
    y_tru=np.array(y_tru)
    loss=sum(sum(abs(y_pre-y_tru)))
    return loss
```


```python
###  lgb
lgb_params = {'boosting_type': 'gbdt',         #默认都会选gbdt
                'objective': 'multiclass',     #多分类问题
                'num_class': 4,             # 4种分类
                'num_leaves': 2 ** 5,        # 由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth 一般特征越多 值越大
                'feature_fraction': 0.8,    # 特征的子抽样率 提高模型训练速度
                'bagging_fraction': 0.8,    #  样本采样率 可以使bagging更快的运行
                'bagging_freq': 4,          # 4轮一次bagging
                'learning_rate': 0.1,       #   步长 过大容易欠拟合 但训练块
                'seed': 2021,             
                'nthread': 28,                 
                'n_jobs':24,
                'verbose': -1,}
folds = StratifiedKFold(n_splits=5,random_state=2021)
train_res_lgb = np.zeros((len(x_train),4),dtype=float)
lgb_res = np.zeros((len(x_test),4),dtype=float)
onehot_encoder = OneHotEncoder(sparse=False)

for i,(train_index,val_index) in enumerate(folds.split(x_train,y_train)):
    print("第----------{}----------次".format(i+1))
    Dtrain = lgb.Dataset(x_train.iloc[train_index],y_train[train_index])
    Dval = lgb.Dataset(x_train.iloc[val_index],y_train[val_index])
    clf = lgb.train(params=lgb_params,train_set=Dtrain,num_boost_round=10000,valid_sets=[Dtrain,Dval],verbose_eval=500,early_stopping_rounds=300)
    train_res_lgb[val_index] = clf.predict(x_train.iloc[val_index],num_iteration=clf.best_iteration)
    val_y = np.array(y_train[val_index]).reshape(-1, 1)
    val_y = onehot_encoder.fit_transform(val_y)
    print("SCORE:",abs_sum(train_res_lgb[val_index],val_y))
    lgb_res += clf.predict(x_test,num_iteration=clf.best_iteration) / folds.n_splits

y_train = onehot_encoder.fit_transform(np.array(y_train).reshape(-1,1))    
print("SCORE:",abs_sum(train_res_lgb,y_train))
```

    第----------1----------次
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 300 rounds
    [500]	training's multi_logloss: 8.46313e-05	valid_1's multi_logloss: 0.0466978
    Early stopping, best iteration is:
    [284]	training's multi_logloss: 0.00138028	valid_1's multi_logloss: 0.0414792
    SCORE: 597.0545258999908
    第----------2----------次
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 300 rounds
    [500]	training's multi_logloss: 9.83338e-05	valid_1's multi_logloss: 0.0396498
    Early stopping, best iteration is:
    [296]	training's multi_logloss: 0.00123954	valid_1's multi_logloss: 0.0365873
    SCORE: 551.2524282384878
    第----------3----------次
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 300 rounds
    [500]	training's multi_logloss: 9.25436e-05	valid_1's multi_logloss: 0.0446953
    Early stopping, best iteration is:
    [280]	training's multi_logloss: 0.00151948	valid_1's multi_logloss: 0.04028
    SCORE: 605.809072153235
    第----------4----------次
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 300 rounds
    [500]	training's multi_logloss: 8.90139e-05	valid_1's multi_logloss: 0.0440625
    Early stopping, best iteration is:
    [273]	training's multi_logloss: 0.00158185	valid_1's multi_logloss: 0.0393608
    SCORE: 600.2245066991883
    第----------5----------次
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 300 rounds
    [500]	training's multi_logloss: 8.66912e-05	valid_1's multi_logloss: 0.0470821
    Early stopping, best iteration is:
    [258]	training's multi_logloss: 0.00190331	valid_1's multi_logloss: 0.0415659
    SCORE: 635.7648245213784
    SCORE: 2990.105357512256
    


```python
res = pd.DataFrame(lgb_res)
submit['label_0']=res[0]
submit['label_1']=res[1]
submit['label_2']=res[2]
submit['label_3']=res[3]
submit.to_csv('submit.csv',index=False)
```
