
# coding: utf-8

# In[1]:


from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import numpy as np


# In[13]:


# Fetching Data and Format it 
data = fetch_movielens(min_rating=4.0)

# print Training and Testing Data
# print(repr(data['train']))
# print(repr(data['test']))
print(data['train'].shape)
data


# In[3]:


# Create Model
model = LightFM(loss='warp')

# Train Model
model.fit(data['train'], epochs=30, num_threads=2)


# In[51]:


def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items  = data['train'].shape
    # n_users = 943
    # n_items = 1682
    
    # generate recommendations for each user we input
    for user_id in user_ids:
        # Movies they already like 
        # Mengambil item_label (judul movie) positif sesuai dengan user_id yg di inputkan
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Movies our model predict they will like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        
        print("User %s"%user_id)
        print("   Know Positives:")
        for x in known_positives[:3]:
            print("      %s" %x)
        
        print("Recomended : ")
        for x in top_items[:3]:
            print("      %s" %x)


# In[52]:


sample_recommendation(model, data, [3])

