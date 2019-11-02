import pandas as pd
import numpy as np
import tensorflow as tf


def load(file):

    import ast
    df = pd.read_csv(file)

    # convert the column values from literal string to dictionary
    df['ltiFeatures'] = df['ltiFeatures'].apply(ast.literal_eval)
    df['stiFeatures'] = df['stiFeatures'].apply(ast.literal_eval)

    return df


def convert_data_to_im(data):

    def find_max_feature(user_n, ):
        max_c = []
        for i in range(user_n):
            user_lti = data.at[i, 'ltiFeatures']
            user_sti = data.at[i, 'stiFeatures']
            c_l = max([int(i) for i in user_lti.keys()])
            if user_sti != {}:
                c_s = max([int(i) for i in user_sti.keys()])
            else:
                c_s = 0
            max_c.append(max(c_l, c_s))
            c = max(max_c)
        return c

    r = int(np.shape(data)[0])  # No. of users
    c = find_max_feature()
    im = np.zeros((r, 2, c))
    for i in range(r):  # iterate users
        user_lti = data.at[i, 'ltiFeatures']
        user_sti = data.at[i, 'stiFeatures']
        for k in user_lti.keys():
            im[i, 0, int(k)-1] = user_lti[k]
        for k in user_sti.keys():
            im[i, 1, int(k)-1] = user_sti[k]
    return im


def ground_truth(data):
    """ Create label from data
    Convert boolean from dataset to labels as the ground truth for machine
    learning

    :param data: DataFrame
    :return: numpy array
    """
    gt = 1 * data.to_numpy()[:, 1]
    return gt


# load all data
training = load("training.csv")
# validation = load("validation.csv")
# interest_topics = pd.read_csv("interest_topics.csv")

# image = convert_data_to_im(training, interest_topics)
G_T = ground_truth(training)

