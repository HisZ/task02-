

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
heartbeatdf = heartbeatdf.reset_index()
print(heartbeatdf.head())
print(heartbeatdf.info())
```

       level_0  level_1                   0
    0        0        0  0.9912297987616655
    1        0        1  0.9435330436439665
    2        0        2  0.7646772997256593
    3        0        3  0.6185708990212999
    4        0        4  0.3796321642826237
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20500000 entries, 0 to 20499999
    Data columns (total 3 columns):
    level_0    int64
    level_1    int64
    0          object
    dtypes: int64(2), object(1)
    memory usage: 469.2+ MB
    None
    


```python
heartbeatdf = heartbeatdf.set_index("level_0")
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
      <th>level_1</th>
      <th>0</th>
    </tr>
    <tr>
      <th>level_0</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.9912297987616655</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.9435330436439665</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.7646772997256593</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.6185708990212999</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0.3796321642826237</td>
    </tr>
  </tbody>
</table>
</div>




```python
heartbeatdf.index.name=None
print(heartbeatdf.head())
print(heartbeatdf.info())
```

       level_1                   0
    0        0  0.9912297987616655
    0        1  0.9435330436439665
    0        2  0.7646772997256593
    0        3  0.6185708990212999
    0        4  0.3796321642826237
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 20500000 entries, 0 to 99999
    Data columns (total 2 columns):
    level_1    int64
    0          object
    dtypes: int64(1), object(1)
    memory usage: 469.2+ MB
    None
    


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
    Int64Index: 20500000 entries, 0 to 99999
    Data columns (total 2 columns):
    time                 int64
    heartbeat_signals    float64
    dtypes: float64(1), int64(1)
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
      <th>time</th>
      <th>heartbeat_signals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.991230</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.943533</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.764677</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.618571</td>
    </tr>
    <tr>
      <th>0</th>
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
#settings = EfficientFCParameters()
settings = MinimalFCParameters()
ext_fea = extract_features(rehpb,column_id="index",column_sort="time",default_fc_parameters=settings)
```

    Feature Extraction: 100%|██████████████████████████████████████████████████████████████| 20/20 [01:33<00:00,  4.68s/it]
    


```python
from tsfresh.utilities.dataframe_functions import impute
#去除非数(NaN)，利用impute函数
ext_fea = impute(ext_fea)
ext_fea
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
      <th>heartbeat_signals__sum_values</th>
      <th>heartbeat_signals__median</th>
      <th>heartbeat_signals__mean</th>
      <th>heartbeat_signals__length</th>
      <th>heartbeat_signals__standard_deviation</th>
      <th>heartbeat_signals__variance</th>
      <th>heartbeat_signals__maximum</th>
      <th>heartbeat_signals__minimum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38.927945</td>
      <td>0.125531</td>
      <td>0.189892</td>
      <td>205.0</td>
      <td>0.229783</td>
      <td>0.052800</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.445634</td>
      <td>0.030481</td>
      <td>0.094857</td>
      <td>205.0</td>
      <td>0.169080</td>
      <td>0.028588</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.192974</td>
      <td>0.000000</td>
      <td>0.103380</td>
      <td>205.0</td>
      <td>0.184119</td>
      <td>0.033900</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42.113066</td>
      <td>0.241397</td>
      <td>0.205430</td>
      <td>205.0</td>
      <td>0.186186</td>
      <td>0.034665</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69.756786</td>
      <td>0.000000</td>
      <td>0.340277</td>
      <td>205.0</td>
      <td>0.366213</td>
      <td>0.134112</td>
      <td>0.999908</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25.524279</td>
      <td>0.041579</td>
      <td>0.124509</td>
      <td>205.0</td>
      <td>0.175176</td>
      <td>0.030687</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>49.344826</td>
      <td>0.326956</td>
      <td>0.240706</td>
      <td>205.0</td>
      <td>0.222915</td>
      <td>0.049691</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>52.710158</td>
      <td>0.336291</td>
      <td>0.257123</td>
      <td>205.0</td>
      <td>0.239443</td>
      <td>0.057333</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>45.128485</td>
      <td>0.267322</td>
      <td>0.220139</td>
      <td>205.0</td>
      <td>0.199813</td>
      <td>0.039925</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>66.477343</td>
      <td>0.340271</td>
      <td>0.324280</td>
      <td>205.0</td>
      <td>0.175118</td>
      <td>0.030666</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>81.041338</td>
      <td>0.595547</td>
      <td>0.395324</td>
      <td>205.0</td>
      <td>0.339943</td>
      <td>0.115561</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>125.884286</td>
      <td>0.842209</td>
      <td>0.614070</td>
      <td>205.0</td>
      <td>0.391304</td>
      <td>0.153119</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>52.067355</td>
      <td>0.336924</td>
      <td>0.253987</td>
      <td>205.0</td>
      <td>0.238822</td>
      <td>0.057036</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>44.122286</td>
      <td>0.271179</td>
      <td>0.215231</td>
      <td>205.0</td>
      <td>0.157393</td>
      <td>0.024773</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>77.671345</td>
      <td>0.566549</td>
      <td>0.378885</td>
      <td>205.0</td>
      <td>0.327372</td>
      <td>0.107173</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>20.595313</td>
      <td>0.080731</td>
      <td>0.100465</td>
      <td>205.0</td>
      <td>0.148004</td>
      <td>0.021905</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>31.183807</td>
      <td>0.164655</td>
      <td>0.152116</td>
      <td>205.0</td>
      <td>0.184128</td>
      <td>0.033903</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17.153033</td>
      <td>0.000000</td>
      <td>0.083673</td>
      <td>205.0</td>
      <td>0.149536</td>
      <td>0.022361</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>31.294642</td>
      <td>0.000000</td>
      <td>0.152657</td>
      <td>205.0</td>
      <td>0.249077</td>
      <td>0.062039</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>23.370098</td>
      <td>0.090995</td>
      <td>0.114000</td>
      <td>205.0</td>
      <td>0.142985</td>
      <td>0.020445</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22.282372</td>
      <td>0.075795</td>
      <td>0.108694</td>
      <td>205.0</td>
      <td>0.152405</td>
      <td>0.023227</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>48.416403</td>
      <td>0.316221</td>
      <td>0.236178</td>
      <td>205.0</td>
      <td>0.209455</td>
      <td>0.043872</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>26.365748</td>
      <td>0.000000</td>
      <td>0.128613</td>
      <td>205.0</td>
      <td>0.204338</td>
      <td>0.041754</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>71.553700</td>
      <td>0.469648</td>
      <td>0.349042</td>
      <td>205.0</td>
      <td>0.293654</td>
      <td>0.086233</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>44.629746</td>
      <td>0.256340</td>
      <td>0.217706</td>
      <td>205.0</td>
      <td>0.191533</td>
      <td>0.036685</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>36.766198</td>
      <td>0.179756</td>
      <td>0.179347</td>
      <td>205.0</td>
      <td>0.166543</td>
      <td>0.027737</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>40.085708</td>
      <td>0.072796</td>
      <td>0.195540</td>
      <td>205.0</td>
      <td>0.230689</td>
      <td>0.053217</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>43.984348</td>
      <td>0.204244</td>
      <td>0.214558</td>
      <td>205.0</td>
      <td>0.156482</td>
      <td>0.024487</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>41.120845</td>
      <td>0.225724</td>
      <td>0.200589</td>
      <td>205.0</td>
      <td>0.193715</td>
      <td>0.037525</td>
      <td>0.997184</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>38.140913</td>
      <td>0.000000</td>
      <td>0.186053</td>
      <td>205.0</td>
      <td>0.240028</td>
      <td>0.057613</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99970</th>
      <td>44.366768</td>
      <td>0.282099</td>
      <td>0.216423</td>
      <td>205.0</td>
      <td>0.192018</td>
      <td>0.036871</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99971</th>
      <td>44.672948</td>
      <td>0.167554</td>
      <td>0.217917</td>
      <td>205.0</td>
      <td>0.227917</td>
      <td>0.051946</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99972</th>
      <td>43.574852</td>
      <td>0.190462</td>
      <td>0.212560</td>
      <td>205.0</td>
      <td>0.242461</td>
      <td>0.058787</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99973</th>
      <td>105.777998</td>
      <td>0.803461</td>
      <td>0.515990</td>
      <td>205.0</td>
      <td>0.416286</td>
      <td>0.173294</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99974</th>
      <td>15.458631</td>
      <td>0.030283</td>
      <td>0.075408</td>
      <td>205.0</td>
      <td>0.163716</td>
      <td>0.026803</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99975</th>
      <td>50.951809</td>
      <td>0.310340</td>
      <td>0.248545</td>
      <td>205.0</td>
      <td>0.192704</td>
      <td>0.037135</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99976</th>
      <td>43.494733</td>
      <td>0.000000</td>
      <td>0.212169</td>
      <td>205.0</td>
      <td>0.283561</td>
      <td>0.080407</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99977</th>
      <td>47.749131</td>
      <td>0.280244</td>
      <td>0.232923</td>
      <td>205.0</td>
      <td>0.183940</td>
      <td>0.033834</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99978</th>
      <td>97.534737</td>
      <td>0.489884</td>
      <td>0.475779</td>
      <td>205.0</td>
      <td>0.209451</td>
      <td>0.043870</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99979</th>
      <td>24.991696</td>
      <td>0.000000</td>
      <td>0.121911</td>
      <td>205.0</td>
      <td>0.202875</td>
      <td>0.041158</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99980</th>
      <td>39.187845</td>
      <td>0.000000</td>
      <td>0.191160</td>
      <td>205.0</td>
      <td>0.226879</td>
      <td>0.051474</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99981</th>
      <td>35.296522</td>
      <td>0.000000</td>
      <td>0.172178</td>
      <td>205.0</td>
      <td>0.225302</td>
      <td>0.050761</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99982</th>
      <td>82.313364</td>
      <td>0.543232</td>
      <td>0.401529</td>
      <td>205.0</td>
      <td>0.345142</td>
      <td>0.119123</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99983</th>
      <td>43.538503</td>
      <td>0.301524</td>
      <td>0.212383</td>
      <td>205.0</td>
      <td>0.204110</td>
      <td>0.041661</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99984</th>
      <td>6.559380</td>
      <td>0.000000</td>
      <td>0.031997</td>
      <td>205.0</td>
      <td>0.101412</td>
      <td>0.010284</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99985</th>
      <td>37.150889</td>
      <td>0.000000</td>
      <td>0.181224</td>
      <td>205.0</td>
      <td>0.263806</td>
      <td>0.069593</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99986</th>
      <td>40.845055</td>
      <td>0.000000</td>
      <td>0.199244</td>
      <td>205.0</td>
      <td>0.230327</td>
      <td>0.053050</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99987</th>
      <td>84.365689</td>
      <td>0.587797</td>
      <td>0.411540</td>
      <td>205.0</td>
      <td>0.333781</td>
      <td>0.111410</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99988</th>
      <td>43.666796</td>
      <td>0.197358</td>
      <td>0.213009</td>
      <td>205.0</td>
      <td>0.155662</td>
      <td>0.024231</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99989</th>
      <td>53.837870</td>
      <td>0.307182</td>
      <td>0.262624</td>
      <td>205.0</td>
      <td>0.230258</td>
      <td>0.053019</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99990</th>
      <td>33.724657</td>
      <td>0.000000</td>
      <td>0.164511</td>
      <td>205.0</td>
      <td>0.244159</td>
      <td>0.059613</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99991</th>
      <td>51.764317</td>
      <td>0.246161</td>
      <td>0.252509</td>
      <td>205.0</td>
      <td>0.195247</td>
      <td>0.038122</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99992</th>
      <td>80.130225</td>
      <td>0.359141</td>
      <td>0.390879</td>
      <td>205.0</td>
      <td>0.177118</td>
      <td>0.031371</td>
      <td>0.983664</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99993</th>
      <td>39.926600</td>
      <td>0.000000</td>
      <td>0.194764</td>
      <td>205.0</td>
      <td>0.260816</td>
      <td>0.068025</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99994</th>
      <td>31.098534</td>
      <td>0.000000</td>
      <td>0.151700</td>
      <td>205.0</td>
      <td>0.228589</td>
      <td>0.052253</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>63.323449</td>
      <td>0.388402</td>
      <td>0.308895</td>
      <td>205.0</td>
      <td>0.211636</td>
      <td>0.044790</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>69.657534</td>
      <td>0.421138</td>
      <td>0.339793</td>
      <td>205.0</td>
      <td>0.199966</td>
      <td>0.039986</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>40.897057</td>
      <td>0.213306</td>
      <td>0.199498</td>
      <td>205.0</td>
      <td>0.200657</td>
      <td>0.040263</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>42.333303</td>
      <td>0.264974</td>
      <td>0.206504</td>
      <td>205.0</td>
      <td>0.164380</td>
      <td>0.027021</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>53.290117</td>
      <td>0.320124</td>
      <td>0.259952</td>
      <td>205.0</td>
      <td>0.194868</td>
      <td>0.037974</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 8 columns</p>
</div>




```python
#然后利用select_features函数过滤特征
#由于特征提取是最小配置 故无须再做特征选择
#ext_fea=select_features(ext_fea,y_train,fdr_level=0.5)
```
