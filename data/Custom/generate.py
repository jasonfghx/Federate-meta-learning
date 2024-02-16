import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math
import pandas as pd

NUM_USER = 15 #可以改這邊 增加client
np.random.seed(2)
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate(alpha, beta):

    dimension = 7
    NUM_CLASS = 2
    
    samples_per_user = (np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 1) *2
    print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]
    X_data=pd.read_csv(fileDirectory+"/anomalydetection_test_x.csv")
    Y_data=pd.read_csv(fileDirectory+"/anomalydetection_test_y.csv")
    for i in range(NUM_USER):
        yy=[]
        for j in range(samples_per_user[i]):
            
            temp=np.random.choice(X_data.shape[0],samples_per_user[i])
            xx=X_data.loc[temp].values
            yy=list(Y_data.loc[temp]['class'])
            # yy=[v[0] for v in yy]
        # print(yy)
        X_split[i] = xx.tolist()
        y_split[i] = yy
        
        print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_split, y_split



def main():


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_path = fileDirectory+"/data/train/mytrain.json"
    test_path = fileDirectory+"/data/test/mytest.json"

    X, y = generate(alpha=0.5, beta=0.5)



    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(NUM_USER, ncols=120):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()

