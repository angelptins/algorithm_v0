#!/usr/bin/env python
# coding: utf-8

# ### Import Packages

# In[1]:


import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.stats import pearsonr, ttest_ind

from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from collections import OrderedDict


# In[21]:


# import warnings
# warnings.filterwarnings('ignore')


# ### Helper Functions

# In[4]:


def loadClassifier(fname_clf):
    '''
    Load scikit-learn model from binary pickle file.
    '''
    with open(fname_clf, 'rb') as f:
        x = pickle.load(f)
    return x


# In[5]:


def STD(data):
    '''
    Input: numpy array (shape: <num_frames,32,32> ) 
    Output: numpy array (shape: <num_frames,32,32> ) normalized by standard deviation of frame values
    '''
    out = []
    for i in range(data.shape[0]):
        matrice = data[i,:,:]
        s = np.std(np.concatenate(matrice))
        out.append(matrice/s)
    return np.array(out, dtype=np.float32)


# In[6]:


def HISTOGRAME(data):
    '''
    Input: numpy array (shape: <num_frames,32,32> ) 
    Output: numpy array (shape: <num_frames,32,32> ) binned into values between 0 and 255
    '''
    out = []
    for i in range(data.shape[0]):
        matrice = data[i,:,:]
        max = np.max(np.concatenate(matrice))
        matrice = 255*matrice/max
        out.append(matrice)
    return np.array(out, dtype=np.float32)


# In[7]:


def BINARY(data):
    '''
    Input: numpy array (shape: <num_frames,32,32> ) 
    Output: numpy array (shape: <num_frames,32,32> ) binarized at threshold > 0
    '''
    out = []
    for i in range(data.shape[0]):
        matrice = data[i,:,:]
        matrice[matrice > 0] = 1
        out.append(m)
    return np.array(out, dtype=np.float32)


# In[8]:


def flatMatrice(data):
    '''
    Input: numpy array (shape: <num_frames,32,32> ) 
    Output: numpy array (shape: <num_frames,1024> )
    '''
    out = []
    for i in range(data.shape[0]):  
        out.append(np.concatenate(data[i,:,:]))
    return np.array(out)


# In[9]:


def dataReduction(data):
    '''
    Input: numpy array (shape: <num_frames,1024> ) of normalized data
    Output: numpy array (shape: <N_CLUSTERS,1024> ) of cluster centers
    '''
    kmeans = KMeans(n_clusters=N_CLUSTERS).fit(data)
    clusters = kmeans.cluster_centers_
    return clusters


# In[10]:


def extractFeatures(data, clusters, keys_clf):
    '''
    Input: 
    - numpy array (shape: <num_frames,1024> ) of normalized data
    - cluster centers from dataReduction function
    - keys of loaded, pre-trained classifier
    Output: 
    - pandas dataframe (shape: <N_CLUSTERS,21> )
    '''
    
    # Basic stats of normalized data.
    data_m = data.mean(axis = 0)
    data_sd = data.std(axis = 0)
       
    # Sort frames (from data var) into two groups based on pearsonr correlation.
    number = np.zeros(N_CLUSTERS)
    sortie = []
    for i in range(data.shape[0]):
        r = []
        for j in range(clusters.shape[0]):
            r_value,pr = pearsonr(data[i], clusters[j])
            r.append(np.abs(r_value))
        r = np.array(r)
        number[np.argmax(r)] = number[np.argmax(r)] + 1
        sortie.append(np.argmax(r))

    # Nonsense, and not used.
    mini = np.argmin(number)    # represent mvt
    maxi = np.argmax(number)    # represent silent
    
    number = 100*number/data.shape[0]
    m_xb = []
    m_yb = []
    
    m_d1 = []
    m_d2 = []
    m_d3 = []
    m_d4 = []
    m_d5 = []
    
    m_q1x = []
    m_q2x = []
    m_q3x = []
    m_q4x = []
    m_q5x = []
    
    m_q1y = []
    m_q2y = []
    m_q3y = []
    m_q4y = []
    m_q5y = []
    
    area = []
    std = []
    moy = []
    
    # Compute statistics for each cluster.
    for jj in range(clusters.shape[0]):
        
        matrice = clusters[jj].reshape(32,32)
        x, y  = ndimage.center_of_mass(matrice)

        xb = x
        yb = y
        
        m_xb.append(xb)
        m_yb.append(yb)
        
        if x < 28 and x > 4:
            if y < 28 and y > 4:
                q1 = np.zeros(9)
                q2 = np.zeros(9)
                q3 = np.zeros(9)
                q4 = np.zeros(9)
                q5 = np.zeros(9)
                
                z = 0
                for j in range(3):
                    for k in range(3):                     
                        q1[int(z)] = (matrice.item((int(x-1+j), int(y-1+k))))
                        q2[int(z)] = (matrice.item((int(x+j-3), int(y+k-4))))
                        q3[int(z)] = (matrice.item((int(x+j+1), int(y+k-4))))
                        q4[int(z)] = (matrice.item((int(x+j-3), int(y+k+2))))
                        q5[int(z)] = (matrice.item((int(x+j+1), int(y+k+2))))
                        z = z+1

                q1 = q1.reshape(3,3).transpose()
                q2 = q2.reshape(3,3).transpose()
                q3 = q3.reshape(3,3).transpose()
                q4 = q4.reshape(3,3).transpose()
                q5 = q5.reshape(3,3).transpose()
                
                x_1, y_1  = ndimage.center_of_mass(q1)
                if np.isnan(x_1) or np.isnan(y_1):
                    x_1 = 0
                    y_1 = 0
                x_2, y_2  = ndimage.center_of_mass(q2)
                if np.isnan(x_2) or np.isnan(y_2):
                    x_2 = 0
                    y_2 = 0
                x_3, y_3  = ndimage.center_of_mass(q3)
                if np.isnan(x_3) or np.isnan(y_3):
                    x_3 = 0
                    y_3 = 0
                x_4, y_4  = ndimage.center_of_mass(q4)
                if np.isnan(x_4) or np.isnan(y_4):
                    x_4 = 0
                    y_4 = 0
                x_5, y_5  = ndimage.center_of_mass(q5)
                if np.isnan(x_5) or np.isnan(y_5):
                    x_5 = 0
                    y_5 = 0

                x_1 = x_1-1
                y_1 = y_1-1
                x_2 = x_2-3
                y_2 = y_2-4
                x_3 = x_3+1
                y_3 = y_3-4
                x_4 = x_4-3
                y_4 = y_4+2
                x_5 = x_5+1
                y_5 = y_5+2

                d_1 = np.sqrt(x_1**2 + y_1**2)
                x = x_2 - x_1
                y = y_2 - y_1
                d_2 = np.sqrt(x**2 + y**2)
                x = x_3 - x_1
                y = y_3 - y_1
                d_3 = np.sqrt(x**2 + y**2)
                x = x_4 - x_1
                y = y_4 - y_1
                d_4 = np.sqrt(x**2 + y**2)
                x = x_5 - x_1
                y = y_5 - y_1
                d_5 = np.sqrt(x**2 + y**2)
            else:
                x_1 = np.nan
                y_1 = np.nan
                x_2 = np.nan
                y_2 = np.nan
                x_3 = np.nan
                y_3 = np.nan
                x_4 = np.nan
                y_4 = np.nan
                x_5 = np.nan
                y_5 = np.nan
                d_1 = np.nan
                d_2 = np.nan
                d_3 = np.nan
                d_4 = np.nan
                d_5 = np.nan
        else:
            x_1 = np.nan
            y_1 = np.nan
            x_2 = np.nan
            y_2 = np.nan
            x_3 = np.nan
            y_3 = np.nan
            x_4 = np.nan
            y_4 = np.nan
            x_5 = np.nan
            y_5 = np.nan
            d_1 = np.nan
            d_2 = np.nan
            d_3 = np.nan
            d_4 = np.nan
            d_5 = np.nan
        
        x1 = x_1
        x2 = x_2
        x3 = x_3
        x4 = x_4
        x5 = x_5
        
        y1 = y_1
        y2 = y_2
        y3 = y_3
        y4 = y_4
        y5 = y_5
        
        d1 = d_1
        d2 = d_2
        d3 = d_3
        d4 = d_4
        d5 = d_5

        q1_area = len(np.where(matrice>0)[0])
        q1_std = matrice.std()
        q1_mean = matrice.mean()
        
        area.append(q1_area)
        std.append(q1_std)
        moy.append(q1_mean)
        
        m_q1x.append(x1)
        m_q1y.append(y1)
        
        m_q2x.append(x2)
        m_q2y.append(y2)
        
        m_q3x.append(x3)
        m_q3y.append(y3)
        
        m_q4x.append(x4)
        m_q4y.append(y4)
        
        m_q5x.append(x5)
        m_q5y.append(y5)
        
        m_d1.append(d1)
        m_d2.append(d2)
        m_d3.append(d3)
        m_d4.append(d4)
        m_d5.append(d5)
    
    area = np.array(area)
    moy = np.array(moy)
    std = np.array(std)

    m_q1x = np.array(m_q1x)
    m_q1y = np.array(m_q1y)
    m_q2x = np.array(m_q2x)
    m_q2y = np.array(m_q2y)
    m_q3x = np.array(m_q3x)
    m_q3y = np.array(m_q3y)
    m_q4x = np.array(m_q4x)
    m_q4y = np.array(m_q4y)
    m_q5x = np.array(m_q5x)
    m_q5y = np.array(m_q5y)
    
    d = OrderedDict()
    for key in keys_clf:
        if key == 'number':
            d['number'] = number
        elif key == 'xb':
            d['xb'] = m_xb
        elif key == 'yb':
            d['yb'] = m_yb
        elif key == 'x1':
            d['x1'] = m_q1x
        elif key == 'y1':
            d['y1'] = m_q1y
        elif key == 'x2':
            d['x2'] = m_q2x
        elif key == 'y2':
            d['y2'] = m_q2y
        elif key == 'x3':
            d['x3'] = m_q3x
        elif key == 'y3':
            d['y3'] = m_q3y
        elif key == 'x4':
            d['x4'] = m_q4x
        elif key == 'y4':
            d['y4'] = m_q4y
        elif key == 'x5':
            d['x5'] = m_q5x
        elif key == 'y5':
            d['y5'] = m_q5y
        elif key == 'd1':
            d['d1'] = m_d1
        elif key == 'd2':
            d['d2'] = m_d2
        elif key == 'd3':
            d['d3'] = m_d3
        elif key == 'd4':
            d['d4'] = m_d4
        elif key == 'd5':
            d['d5'] = m_d5
        elif key == 'ga':
            d['ga'] = area
        elif key == 'gm':
            d['gm'] = moy
        elif key == 'gs':
            d['gs'] = std

    d = pd.DataFrame.from_dict(d)
    d = d.dropna()

    return d


# In[11]:


def classifier(features, clf):
    '''
    Input: 
    - pandas dataframe (shape: <N_CLUSTERS,21> ) from extractFeatures function
    - pre-trained classifier
    Output: 
    - list of predictions (shape: <N_CLUSTERS,1>) -- each cluster gets classified by the clf??
    '''
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features) # Why do this?
    y_prediction = clf.predict(X)
    return y_prediction


# In[12]:


def correction(prediction, features):
    '''
    Input: 
    - list of predictions (shape: <N_CLUSTERS,1>) -- each cluster gets classified by the clf??
    - pandas dataframe (shape: <N_CLUSTERS,21> ) from extractFeatures function
    Output: 
    - integer (0 for non-CS, 1 for CS)
    '''

    d = {}
    d['number'] = features['number'].values
    d['pred'] = np.array(prediction)
    df =  pd.DataFrame.from_dict(d)
    zero = df[df['pred']==0]['number'].sum()
    one = df[df['pred']==1]['number'].sum()
    if zero > one:
        return 0
    else:
        return 1


# In[13]:


def get_pred_from_fname(fname_data, 
                        fname_clf,
                        transpose=False,
                        num_iters=20):
    
    datalist = None

#     ### ARNAUD VERSION
#     # Read in data.
#     if '.txt' in fname_data:
#         print('Reading tabular data from tab-separated text file...')
#         datalist = pd.read_csv(fname_data, sep='\t', encoding="ISO-8859-1", low_memory=False) 
#         datalist2 = datalist.values[11:, 1:]
#     elif '.csv' in fname_data:
#         print('Reading tabular data from comma-separated text file...')
#         datalist = pd.read_csv(fname_data, sep=',', encoding="ISO-8859-1", low_memory=False)    
#         datalist2 = datalist.values[11:, 1:]

    ### PT VERSION
    # Read in data.
    print('Reading in data...')
    datalist = pd.read_csv(fname_data, index_col=0).T
    datalist2 = datalist.values
    print('Done.')
    
    print('Reshaping data...')
    if datalist.shape[1] < 3000:
        print('Not enough data... exiting.')
        return
    else:
        datalist_tmp = []
        if transpose:
            for tf in range(datalist2.shape[1]):
                d = datalist2[:, tf].reshape(32,32).astype(np.float64)
                d = d.T 
                d = d[:,::-1]
                datalist_tmp.append(d)
        else:
            for tf in range(datalist2.shape[1]):
                d = datalist2[:, tf].reshape(32,32).astype(np.float64)
                d = d[:,::-1]
                datalist_tmp.append(d)
        datalist3 = np.array(datalist_tmp)
    print('Done.')
    
    print('Normalizing data...')
    if 'STD' in fname_clf:
        print('Method: STD')
        datalist4 = STD(datalist3)
    elif 'BINARY' in fname_clf:
        print('Method: BINARY')
        datalist4 = BINARY(datalist3)
    elif 'HISTOGRAME' in fname_clf:
        print('Method: HIST')
        datalist4 = HISTOGRAME(datalist3)
    print('Done.')
    
    print('Iterating with different starting points...')
    
    maxi = datalist4.shape[0]-3000
    clf, keys = loadClassifier(fname_clf)
    
    preds_corrected_all = []
    for i in tqdm( range(num_iters) ):
    
        start = np.random.randint(0,high=maxi)

        matrix3000 = datalist4[ start:start+3000, : ]
        matrix3000_flat = flatMatrice(matrix3000)

        clusters = dataReduction(matrix3000_flat)

        features = extractFeatures(matrix3000_flat, clusters, keys)

        prediction = classifier(features, clf)
        
        prediction_corrected = correction(prediction, features)
    
        preds_corrected_all.append(prediction_corrected)
    
    
    print('Done.')
    
    print('preds after correction: {}'.format(preds_corrected_all))
    
    if np.mean(preds_corrected_all) < THRESHOLD:
        return False
    else:
        return True


# In[ ]:





# ### Start Here

# In[15]:


fname_data = './sample_data.csv'


# In[16]:


fname_clf = './models/DecisionTreeClassifier_STD.pkl'


# In[17]:


N_CLUSTERS = 2


# In[18]:


THRESHOLD = 0.55


# In[19]:


pred = get_pred_from_fname(fname_data, fname_clf, transpose=False, num_iters=20)


# In[24]:


print('***')
print('fname_data: {}'.format(fname_data))
print('model: {}'.format(fname_clf))
print('pred: {}'.format(pred))
print('***')


# In[ ]:





# In[22]:


### Don't mind this for now.


# In[23]:


# df_gt = pd.read_csv('./GM_NICU_Excel_GMA_output_withFNAMES.csv')


# In[24]:


# fname_data.split('/')[-1]


# In[25]:


# for i, row in df_gt.iterrows():
#     if fname_data.split('/')[-1] in row['fnames_filt']:
#         print(row)


# In[26]:


# preds_for_df = []
# for _, row in df_gt.sample(1).iterrows():
#     fname_data = './{}'.format(row['fnames_filt'])
#     print(fname_data)
#     pred = get_pred_from_fname(fname_data, fname_clf, transpose=False)
#     preds_for_df.append(pred)


# In[ ]:




