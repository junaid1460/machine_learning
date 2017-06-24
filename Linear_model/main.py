
# coding: utf-8

# In[ ]:


import pandas as pd
import tensorflow as tf
import io
import requests


# # read data and fix errors
# the data read has some minor errors like string 'false' instead of boolean False
# let's fix them

# In[2]:



#read training data set

def read_train_data():
    return pd.read_csv('./train.csv')
# normalize column values
def fix_errors(data):
    data['disable_communication'] = data['disable_communication'].replace(['false'],False)
    data['disable_communication'] = data['disable_communication'].replace(['true'],True)
    print('replaced \'false\' with False and \'true\' with True') 
    return data


# In[3]:


#exchange rates from yahoo

def read_exchange_rates_json():
    url = 'https://finance.yahoo.com/webservice/v1/symbols/allcurrencies/quote?format=json'
    try:
        return pd.read_json(io.StringIO(requests.get(url).content.decode('utf-8')))['list']['resources']
    except:
        print('couldn\'t fetch data from yahoo, loading local data')
        return pd.read_json('./currencyrates.json')['list']['resources']
        

# make dictionary of required field
def get_exchange_rates_dataframe(data):
    rates = []
    for rate in data:
        dt = rate['resource']['fields']
        name = dt['name']
        if(name[0:3]!='USD'):
            continue
            
        if(len(name) < 6):
            rates = rates + [{'to':name,'price' : float(dt['price'])  }]
            continue
            
        _from,_to = name.split('/')
        rates = rates + [{'to':_to,'price' : float(dt['price'])  }]
    
    return pd.DataFrame.from_records(rates)

def get_price(code,data):
    return data[data.to == code ].price
# data frame from dictionary list


# # data normalization
# the goal price is represented using many currencies. we convert all the currrency to US Dollars by grabbing exchange rate from Yahoo and diving all currencies  by it's exhange rate.

# In[4]:


def normalize_price(data,ex_data):
    for index,dt in data.iterrows():
        price = float(dt['goal']) / get_price(dt['currency'],ex_data) 
        data.set_value(index,'goal',price)
    print('price normalization complete')
    return data

#modification to project name and description
def get_ratio(name):
    sc = 1
    ab = 1
    for c in name:
        if c.isalpha():
            ab += 1
        else:
            sc += 1
    return sc/ab
            

def normalize_name_and_desc(data):
    for index,dt in data.iterrows():
        name_ratio = get_ratio(str(dt['name']))
        desc_ratio = get_ratio(str(dt['desc']))
        data.set_value(index,'name',name_ratio)
        data.set_value(index,'desc',desc_ratio)
    print('name and desc normalization complete')
    return data
    
def normalize_deadline(data):
    for index,dt in data.iterrows():
        deadline = dt['deadline'] - dt['launched_at']
        data.set_value(index,'deadline',deadline)
    print('deadline normalization complete')
    return data
    
def normalize(data,ex_data):
    #data = normalize_deadline(data)
    #data = normalize_name_and_desc(data)
    data = normalize_price(data,ex_data)
    return data
    
    


# In[5]:


'''
work flow : 
 1. Read training data
 2. fix_errors
 3. Read exchange rates json
 4. Get dataframe of exchange rates
 5. Normalize
 
'''
train_data = read_train_data()
print('read train data')
train_data = fix_errors(train_data)
print('errors fixed')
ex_rates = read_exchange_rates_json()
print('read exchange rates json')
ex_rates = get_exchange_rates_dataframe(ex_rates)
print('dataframe is generated from json data')
train_data = normalize(train_data,ex_rates)



# In[6]:


import numpy as np


# In[7]:


#linear regression



# print('making lists')
# names  = train_data['name'].tolist()
# _name = tf.contrib.layers.real_valued_column("name")
# print('names')
# desc   = train_data['desc'].tolist()
# _desc = tf.contrib.layers.real_valued_column("desc")

print('description')
goal   = train_data['goal'].tolist()
_goal = tf.contrib.layers.real_valued_column("goal")

print('goal')
dis    = train_data['disable_communication'].tolist()
_dis = tf.contrib.layers.real_valued_column("discomm")

print('disable communication')
dead   = train_data['deadline'].tolist()
_dead = tf.contrib.layers.real_valued_column("deadline")

print('deadline')
bac    = train_data['backers_count'].tolist()
_bac = tf.contrib.layers.real_valued_column("backers")

print('backers')

length = len(goal)
steps  = length
all_xs     = []
print('list imported')

for i in range(0,length):
    all_xs.append([goal[i],dis[i],dead[i],bac[i]])
all_ys = train_data['final_status'].tolist()



print('array created')


# In[8]:


import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics


# In[11]:


# replace True , False with 1,0
for i in range(len(all_xs)):
    if all_xs[i][1] == True:
        all_xs[i][1] = 1
    elif all_xs[i][1] == False:
        all_xs[i][1] = 0
day = 60 * 60 * 60 * 24
# converts seconds into days
for i in range(len(all_xs)):
    all_xs[i][2] = all_xs[i][2]/day
    
    


# In[12]:


#training data
#x
target = np.array(all_ys)
#y
data = np.array(all_xs)


# In[ ]:


#tensorflow linear classifier 
classifier = skflow.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=6)],
                                     n_classes = 4,
                                     optimizer=tf.train.FtrlOptimizer(
                                                  learning_rate=0.05,
                                                  l1_regularization_strength=0.001
                                                ),
                                     model_dir = './tmp'
                                    )


# In[ ]:


import shutil

#remove existing data

classifier.fit(data,target)


# In[ ]:




