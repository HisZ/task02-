

```python
import pandas as pd
import numpy as np 
from tsfresh import extract_features
from tsfresh.feature_extraction  import MinimalFCParameters,EfficientFCParameters
```


```python
test = pd.read_csv("testA.csv")
train = pd.read_csv("train.csv")
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
heartbeatdf = train["heartbeat_signals"].str.split(",",expand=True)
heartbeatdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
      <th>200</th>
      <th>201</th>
      <th>202</th>
      <th>203</th>
      <th>204</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.9912297987616655</td>
      <td>0.9435330436439665</td>
      <td>0.7646772997256593</td>
      <td>0.6185708990212999</td>
      <td>0.3796321642826237</td>
      <td>0.19082233510621885</td>
      <td>0.040237131594430715</td>
      <td>0.02599520771717858</td>
      <td>0.03170886048677242</td>
      <td>0.06552357497104398</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9714822034884503</td>
      <td>0.9289687459588268</td>
      <td>0.5729328050711678</td>
      <td>0.1784566262750076</td>
      <td>0.1229615224365985</td>
      <td>0.13236021729815928</td>
      <td>0.09439236984499814</td>
      <td>0.08957535516351411</td>
      <td>0.030480606866741047</td>
      <td>0.04049936195430977</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.9591487564065292</td>
      <td>0.7013782792997189</td>
      <td>0.23177753487886463</td>
      <td>0.0</td>
      <td>0.08069805776387916</td>
      <td>0.12837603937503544</td>
      <td>0.18744837555079963</td>
      <td>0.28082571505275855</td>
      <td>0.3282610568488903</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9757952826275774</td>
      <td>0.9340884687738161</td>
      <td>0.6596366611990001</td>
      <td>0.2499208267606008</td>
      <td>0.23711575621286213</td>
      <td>0.28144491730834825</td>
      <td>0.2499208267606008</td>
      <td>0.2499208267606008</td>
      <td>0.24139674778512604</td>
      <td>0.2306703464848836</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.055816398940721094</td>
      <td>0.26129357194994196</td>
      <td>0.35984696254197834</td>
      <td>0.43314263962884686</td>
      <td>0.45369772898632504</td>
      <td>0.49900406742109477</td>
      <td>0.5427959768500487</td>
      <td>0.6169044962835193</td>
      <td>0.6766958323316207</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 205 columns</p>
</div>




```python
heartbeatdf = heartbeatdf.stack()
heartbeatdf.head()
```




    0  0    0.9912297987616655
       1    0.9435330436439665
       2    0.7646772997256593
       3    0.6185708990212999
       4    0.3796321642826237
    dtype: object




```python
heartbeatdf.index
```




    Int64Index([    0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,
                ...
                99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999,
                99999],
               dtype='int64', length=20500000)




```python
heartbeatdf = heartbeatdf.reset_index()
print(heartbeatdf.head())
print(heartbeatdf.info())
```

       index  time  heartbeat_signals
    0      0     0           0.991230
    1      0     1           0.943533
    2      0     2           0.764677
    3      0     3           0.618571
    4      0     4           0.379632
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20500000 entries, 0 to 20499999
    Data columns (total 3 columns):
    index                int64
    time                 int64
    heartbeat_signals    float64
    dtypes: float64(1), int64(2)
    memory usage: 469.2 MB
    None
    


```python
heartbeatdf = heartbeatdf.set_index("level_0")
heartbeatdf.head()
```


```python
heartbeatdf.index.name=None
print(heartbeatdf.head())
print(heartbeatdf.info())
```


```python
heartbeatdf.rename(columns={"level_1":"time",0:"heartbeat_signals"},inplace=True)
```


```python
heartbeatdf["heartbeat_signals"] = heartbeatdf["heartbeat_signals"].astype("float64")
```


```python
heartbeatdf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20500000 entries, 0 to 20499999
    Data columns (total 3 columns):
    index                int64
    time                 int64
    heartbeat_signals    float64
    dtypes: float64(1), int64(2)
    memory usage: 469.2 MB
    


```python
heartbeatdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>time</th>
      <th>heartbeat_signals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.991230</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0.943533</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0.764677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>0.618571</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>0.379632</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train = train["label"]
```


```python
rehpb = heartbeatdf.reset_index()
rehpb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_0</th>
      <th>index</th>
      <th>time</th>
      <th>heartbeat_signals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.991230</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.943533</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0.764677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0.618571</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0.379632</td>
    </tr>
  </tbody>
</table>
</div>




```python
#settings = EfficientFCParameters()
settings = MinimalFCParameters()
ext_fea = extract_features(rehpb,column_id="index",column_sort="time",default_fc_parameters=settings)
```

    Feature Extraction: 100%|██████████████████████████████████████████████████████████████| 20/20 [01:55<00:00,  5.80s/it]
    


```python
ext_fea
```
