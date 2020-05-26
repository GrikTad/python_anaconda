# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:23:08 2020

@author: Grik
"""

'''import telepot
token='807698753:AAEMHxjvCZxLdmUT5TRmDGupwyzHd0UESxs'
TelegramBot=telepot.Bot(token)
print(TelegramBot.getMe())
TelegramBot.getUpdates()
TelegramBot.getChat('807698753')'''
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=150,batch_size=10)