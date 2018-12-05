import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
#movies with rating 4.0 or higher
data = fetch_movielens(min_rating=4.0) # 100k movie rating from 1k users on 1700 movies - each user rated at least 20 movies 1-5

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss='warp') # warp = Weighted Approximate-Rank Pairwise - https://medium.com/@gabrieltseng/intro-to-warp-loss-automatic-differentiation-and-pytorch-b6aa5083187a

#train model
model.fit(data['train'], epochs=30)

def sample_recommendation(model, data, user_ids):
  #number of users and movies in training data
  n_users, n_items = data['train'].shape

  #generate recommendations for each user we input
  for user_id in user_ids:
    #movies they already like
    known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

    #movies our model predicts they will like
    scores = model.predict(user_id, np.arange(n_items))

    #rank them in order of most liked to least
    top_items = data['item_labels'][np.argsort(-scores)]

    #print out the results
    print('User {}:'.format(user_id))
    print('     Known positives:')
    for kp in known_positives[:3]:
      print('         {}'.format(kp))
    
    print('     Recommended:')
    for ti in top_items[:3]:
      print('         {}'.format(ti))

sample_recommendation(model, data, [3, 25, 450])