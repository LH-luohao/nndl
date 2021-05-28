# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 08:33:36 2021

@author: 罗浩
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei' # 中文乱码
plt.rcParams['axes.unicode_minus']=False # 负号无法正常显示

import chinese_calendar
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


#数据读取
df = pd.read_csv(r'F:\神经网络\数据\nanning_line1.csv', encoding='utf-8', parse_dates=['date'])
weather = pd.read_csv(r'F:\神经网络\数据\nanning_weather.csv', encoding='gbk',parse_dates=['date'])

#数据探索
#df.shape
#weather.shape
#
#df.info()
#weather.info()

#测试数据输出展示
#df.head()
#print(df)
#weather.head()
#print(weather)

#数据合并
df = pd.merge(df,weather,on='date')
#df.head()
#print(df)

#数据预处理

#查看数据分布
#plt.figure(figsize=(15,5))
#plt.plot(df.iloc[:,0],df.iloc[:,1]/10000, label='地铁线')
#plt.legend()

#查看异常值
plt.figure(figsize=(10,5))
p = df.boxplot(return_type='dict')         #画箱线图，直接使用DataFrame的方法
x = p['fliers'][0].get_xdata()               # 'flies'即为异常值的标签
y = p['fliers'][0].get_ydata()
y.sort()
for i in range(len(x)):
  if i>0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.10/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.10,y[i]))

#找出异常数据
df[df['p_flow'].isin(y)]

#删除异常数据
df.drop(df[df['p_flow'].isin(y)].index, inplace=True)

#查看删除效果
#plt.figure(figsize=(10,5))
#p = df.boxplot(return_type='dict')         #画箱线图，直接使用DataFrame的方法
#x = p['fliers'][0].get_xdata()               # 'flies'即为异常值的标签
#y = p['fliers'][0].get_ydata()
#y.sort()
#for i in range(len(x)):
#  if i>0:
#    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.10/(y[i]-y[i-1]),y[i]))
#  else:
#    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.10,y[i]))


#特征构造 是否节假日期
df['is_holiday'] = df['date'].apply(lambda x: 1 if chinese_calendar.is_holiday(x)==True else 0 )
df.index = range(df.shape[0])  # 恢复索引
#增加月份
df['month'] = df['date'].map(lambda x: x.month)
#增加星期
df['dayofweek']=df['date'].dt.dayofweek +1 # +1 之后数字几就代表星期几
#增加前一天
df['pre_date_flow'] = df.loc[:,['p_flow']].shift(1)
#5日，10日移动平均
df['MA5'] = df['p_flow'].rolling(5).mean()
df['MA10'] = df['p_flow'].rolling(10).mean()

#删除节假日
#节假日对于日常规律的客流数据而言，是异常值，所以进行删除
holiday_list =[ '2019-01-01'
               ,'2019-02-04','2019-02-05','2019-02-06','2019-02-07','2019-02-08','2019-02-09','2019-02-10'
               ,'2019-04-05','2019-04-06','2019-04-07'
               ,'2019-05-01','2019-05-02','2019-05-03','2019-05-04'
               ,'2019-06-07','2019-06-08','2019-06-09'
               ,'2019-09-13','2019-09-14','2019-09-15'
               ,'2019-10-01','2019-10-02','2019-10-03','2019-10-04','2019-10-05','2019-10-06','2019-10-07'
              ]
df = df[df['date'].isin(holiday_list) == False]
df.index = range(df.shape[0])
df.head()

#删除缺失值
df.dropna(inplace=True)

#获取所有特征变量
#feature = df.drop(['p_flow'],axis=1)
#得到相关性矩阵
#corr = feature.corr()
#特征矩阵热力图可视化
#plt.figure(figsize=(10,6))
#ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, linewidths=0.2, cmap="RdYlGn",annot=True)
#plt.title("变量间相关系数")

#特征相关性分析，是为了查看特征之间是否存在多重共线性
#如果有多重共线性的话，就要对相关性特别高的特征进行有选择的删除
#从热力图的结果来看，MA5和MA10的相关性是最高的，但也可以接受，不需要对特征进行删除

#目标相关性分析
#df_onehot = pd.get_dummies(df)
#df_onehot.head()
#可视化展示
#plt.figure(figsize=(20,4))
#df_onehot.corr()['p_flow'].sort_values(ascending=False).plot(kind='bar')
#plt.title('人流与其他变量相关性')

#客流和前一天的客流（pre_date_flow）, 是否为节假日（is_holiday）, 周期（dayofweek）, 
#前10日平均客流（MA10），前5日平均客流（MA5） 是有相关性的，并且是正相关。
#从天气情况来看，客流和低温（top_temp_12C），大风 (wind_东南风5级) ， 高温（top_temp_35C）成正相关。
#和舒服的气温（weather晴天），舒适的温度（bot_temp_23C），成负相关。
#说明恶劣天气的时候，选择乘坐地铁的人比较多，而天气好的时候，大家出行的可选性比较多。
#从星期来看，和星期五，星期六，星期日成正相关，和周一，周二，周三，周四成负相关。
#说明周末的客流数更多，而工作日的客流更少。


#模型搭建
##构建特征值X 和目标值 Y 
x = df[['is_holiday','month','dayofweek','pre_date_flow','MA5','MA10']]
y = df['p_flow']
x = np.array(x)
y = np.array(y)

#划分训练集和测试集
#需要注意的是，这里的模型是一个时间序列问题，
#需要用前面的时间数据来预测后面的时间序列问题，
#故在此用前90%作为训练集，后10%作为测试集。
#而不能用train_test_split方法，对训练集和测试集进行随机划分。
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x = scaler_x.fit_transform(x)#自变量和因变量分别归一化
y = scaler_y.fit_transform(np.reshape(y,(len(y),1)))
x_length = x.shape[0]
split = int(x_length*0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# LSTM
model = Sequential()
model.add(LSTM(32, input_dim=1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

# 预测
predict = model.predict(x_test)
plt.figure(figsize=(15,6))
plt.title('预测结果图')
y_test = scaler_y.inverse_transform(np.reshape(y_test,(len(y_test),1)))
predict = scaler_y.inverse_transform(predict)
plt.plot(y_test.ravel(),label='真实值')
plt.plot(predict,label='预测值')
plt.xticks([])
plt.legend()
plt.show()

#model.save('lstm(32).h5')