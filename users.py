import random
import pandas as pd
from sklearn.model_selection import train_test_split
from RecSystem import *

def generate_user(data, n, size):
    l = []
    for i in range(n):
        sample = data.sample(size)
        sample['user_id'] = i
        l.append(sample)
    return pd.concat(l).reset_index(drop=True)


def show_popular(users, data):
    pm = popularity_recommender_py()
    pm.create(data, 'user_id', 'song')

    user_id = random.randint(0,len(users))
    return pm.recommend(user_id).iloc[:, 1:]

def similarity_rec(user_id, data):
    is_model = item_similarity_recommender_py()
    is_model.create(data, 'user_id', 'song')
    return is_model.recommend(user_id)


def similar_item(choice, user_id, data):
    is_model = item_similarity_recommender_py()
    is_model.create(data, 'user_id', 'song')
    return is_model.get_similar_items([str(choice)])

def main():
    data = pd.read_csv('spotify_songs.csv')
    df = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
               'duration_ms', 'playlist_genre', 'playlist_subgenre']]
    df['song'] = data['track_name'].map(str) + " - " + data['track_artist']
    user_df = generate_user(df, 100, 1000)
    users = user_df['user_id'].unique()
    train_data, test_data = train_test_split(user_df, test_size=0.20, random_state=0)
    print(train_data[train_data['user_id']==2][['song', 'user_id']])
    print(similarity_rec(2, train_data))
main()