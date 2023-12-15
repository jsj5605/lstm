# lstm
- Sequential 데이터: 데이터의 순서 정보가 중요한 데이터셋
- 입력 데이터의 sequence가 길수록 Gradient Vanishing으로 초기 Sequence에 대한 학습이 안되는 문제가 RNN의 고질적인 문제
- 이런 Simple RNN의 문제 모델 구조로 해결한 모델이 LSTM
## 1. lstm 주가예측 - import
```python
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchinfo

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split  

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```
## 2. 데이터 수집
```python
# Data loading + EDA
df = pd.read_csv("datasets/005930.ks.csv")
df.shape

output:
(5977, 7)
```
```python
df.info()

output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5977 entries, 0 to 5976
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Date       5977 non-null   object 
 1   Open       5977 non-null   float64
 2   High       5977 non-null   float64
 3   Low        5977 non-null   float64
 4   Close      5977 non-null   float64
 5   Adj Close  5977 non-null   float64
 6   Volume     5977 non-null   int64  
dtypes: float64(5), int64(1), object(1)
memory usage: 327.0+ KB
```
```python
df.head()

output:
	Date	      Open	High	Low	Close	Adj Close	Volume
0	2000-01-04	6000.0	6110.0	5660.0	6110.0	4514.579590	74195000
1	2000-01-05	5800.0	6060.0	5520.0	5580.0	4122.971680	74680000
2	2000-01-06	5750.0	5780.0	5580.0	5620.0	4152.528320	54390000
3	2000-01-07	5560.0	5670.0	5360.0	5540.0	4093.416748	40305000
4	2000-01-10	5600.0	5770.0	5580.0	5770.0	4263.359375	46880000
```
```python
# 결측치 확인
df.isnull().sum()

output:
Open         0
High         0
Low          0
Close        0
Adj Close    0
Volume       0
dtype: int64
```
```python
# 시가, 종가의 흐름을 선그래프로 보기
df[['Open', "Close"]][:50].plot(figsize=(20, 5), alpha=0.5, marker=".");
```
![asdf](https://github.com/jsj5605/lstm/assets/141815934/c8653de9-4fe5-480f-a4a6-975e9afeabb6)

## 3. X, y 나누기
- X (input) feature 구성: open, high, low, close, adj close, volumn
- y (output) : close
```python
y_df = df['Close'].to_frame() # (총데이터수, 1)
X_df = df
X_df.shape, y_df.shape

output:
((5977, 6), (5977, 1))
```
```python
y_df.head()

output:
	          Close
Date	
2000-01-04	6110.0
2000-01-05	5580.0
2000-01-06	5620.0
2000-01-07	5540.0
2000-01-10	5770.0
```
```python
X_df.head()

output:
            	Open	High	Low	Close	Adj Close	Volume
Date						
2000-01-04	6000.0	6110.0	5660.0	6110.0	4514.579590	74195000
2000-01-05	5800.0	6060.0	5520.0	5580.0	4122.971680	74680000
2000-01-06	5750.0	5780.0	5580.0	5620.0	4152.528320	54390000
2000-01-07	5560.0	5670.0	5360.0	5540.0	4093.416748	40305000
2000-01-10	5600.0	5770.0	5580.0	5770.0	4263.359375	46880000
```
## 4. 데이터 전처리
- feature scaling : feature 간의 scale(단위)을 맞추는 작업
- X: Standard Scaling (평균: 0, 표준편차: 1)
- y: MinMax Scaling (최소: 0, 최대: 1) -> X의 scale과 비슷한 값으로 변환
```python
# 객체생성 -> fit() -> transform()
X_scaler = StandardScaler()
y_scaler = MinMaxScaler()

X = X_scaler.fit_transform(X_df)
y = y_scaler.fit_transform(y_df)

print(type(X), type(y))
X.shape, y.shape

output:
<class 'numpy.ndarray'> <class 'numpy.ndarray'>
((5977, 6), (5977, 1))
```
## 5. Sequential Data 구성
- X: 50일치 데이터(ex:1일 ~ 50일), y: 51일째 주가. (ex: 51일)
- 50일의 연속된 주식데이터를 학습하여 51일째 주가를 예측한다.
- X의 한개의 데이터가 50일치 주가데이터가 된다.
![asdf4t](https://github.com/jsj5605/lstm/assets/141815934/c6a05aee-f6e5-4c90-bd72-a9b97b99bc15)

```python
timestep = 50 # sequence length
data_X = [] # X 데이터를 모을 리스트 X: (50, 6)
data_y =[] # y 값을 모을 리스트

for i in range(0, y.size - timestep): # 총 개수 - seq_length: 이 이후 반복시에는 남은 데이터가 51개가 안되어 데이터구성이 안됨
    # X: 0 ~ 50-1, y: 50 (1씩 증가)
    _X = X[i:i+timestep]
    _y = y[i+timestep]
    data_X.append(_X)
    data_y.append(_y)
```
```python
timestep = 50 # sequence length
data_X = [] # X 데이터를 모을 리스트 X: (50, 6)
data_y =[] # y 값을 모을 리스트

for i in range(0, y.size - timestep): # 총 개수 - seq_length: 이 이후 반복시에는 남은 데이터가 51개가 안되어 데이터구성이 안됨
    # X: 0 ~ 50-1, y: 50 (1씩 증가)
    _X = X[i:i+timestep]
    _y = y[i+timestep]
    data_X.append(_X)
    data_y.append(_y)

len(data_X), len(data_y)

output:
(5927, 5927)
```
## 6. train, test set 분리
```python
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
np.shape(X_train), np.shape(X_test)

output:
((4741, 50, 6), (1186, 50, 6))
```
## 7. dataset, dataloader 구성
```python
# Tensor 변환: 
# X_train: list -> ndarray-> torch.Tensor
X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
```
```python
# dataset 생성 -> raw 데이터: tensor -> tensordataset
trainset = TensorDataset(X_train_tensor, y_train_tensor)
testset = TensorDataset(X_test_tensor, y_test_tensor)

print("데이터개수:", len(trainset), len(testset))

output:
데이터개수: 4741 1186
```
```python
# dataloader 생성
trainloader = DataLoader(trainset, batch_size=200, shuffle=True, drop_last=True)
testloader = DataLoader(testset, batch_size=200)

print("에폭당 step수:", len(trainloader), len(testloader))

output:
에폭당 step수: 23 6
```













