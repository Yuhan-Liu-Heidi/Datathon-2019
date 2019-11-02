import pandas as pd
import numpy as np
import ast
import tensorflow as tf


def load(file):
    """ Load dataset
    Provided by Datathon, 2019 Duke

    :param file: string containing the name of the dataset csv file
    :return: DataFrame
    """

    df = pd.read_csv(file)

    df['ltiFeatures'] = df['ltiFeatures'].apply(ast.literal_eval)
    df['stiFeatures'] = df['stiFeatures'].apply(ast.literal_eval)

    return df


def convert_data_to_im(data):
    """ Convert dataset to image
    Creating 1 image per user to represent their long and short term interest.
    Rows: row 0 is the long term interest, row 1 is the short term interest
    Cols: # of col correspond to the topic_id
    Image dimension: # of user * 2 * # of topics

    :param data: DataFrame
    :return: numpy array
    """

    def find_max_feature(user_n, feature_name):
        max_c = []
        for j in range(user_n):
            user_ti = data.at[j, feature_name]
            try:
                m_i = max([int(x) for x in user_ti.keys()])
            except ValueError:
                m_i = 0
            max_c.append(m_i)

        return max(max_c)

    def fill_normalized_value(n, feature_name):
        user_i = data.at[i, feature_name]
        try:
            w_norm = 1 / sum([float(x) for x in user_i])
        except ZeroDivisionError:
            return
        for k in user_i.keys():
            im[i, n, int(k) - 1] = w_norm * user_i[k]
        return

    r = int(np.shape(data)[0])  # No. of users
    c_l = find_max_feature(r, 'ltiFeatures')
    c_s = find_max_feature(r, 'stiFeatures')
    im = np.zeros((r, 2, max(c_l, c_s)))

    for i in range(r):  # iterate users
        fill_normalized_value(0, 'ltiFeatures')
        fill_normalized_value(1, 'stiFeatures')

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
validation = load("validation.csv")
interest_topics = pd.read_csv("interest_topics.csv")

image = convert_data_to_im(training)
G_T = ground_truth(training)
