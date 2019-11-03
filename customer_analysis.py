from data_inspection import load
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt


def categorize_feature(f):
    all_topics = {}
    feature = f.to_numpy()
    for x in range(len(feature[:, 1])):
        topic = feature[x, 1].split('/')[1]
        index = feature[x, 0]
        if topic not in all_topics.keys():
            all_topics[topic] = []
        if not all_topics[topic]:
            all_topics[topic] = [index]
        else:
            all_topics[topic].append(index)
    return all_topics


def label_customer(user_id, data, all_topics):
    d = data.to_numpy()
    index = [i for i, x in enumerate(d[:, 0]) if x == user_id][0]
    interest = d[index, 2]  # dict
    label = 'No interest'
    interest_level = 0
    for k in all_topics.keys():
        value = 0
        for x in all_topics[str(k)]:
            try:
                item = interest[str(x)]
            except KeyError:
                continue
            value += item
        if value > interest_level:
            label = k
            interest_level = value
    return user_id, label


def cluster_user(data, all_topics):
    d = data.to_numpy()
    user_n = int(np.shape(d)[0])
    conversion = {}
    customer_n = {}
    conversion_rate = {}
    for i in range(user_n):
        [_, label] = label_customer(d[i, 0], data, all_topics)
        if not label == 'No interest':
            if label not in conversion.keys():
                conversion[label] = 0
            conversion[label] += 1 * d[i, 1]
            if label not in customer_n.keys():
                customer_n[label] = 0
            customer_n[label] += 1
    for x in all_topics.keys():
        try:
            conversion_rate[x] = conversion[x]/customer_n[x]
        except ZeroDivisionError:
            continue
        except KeyError:
            conversion_rate[x] = 0
    return conversion_rate


training = load("training.csv")
interest_topics = pd.read_csv("interest_topics.csv")

all_t = categorize_feature(interest_topics)
# [_, Label] = label_customer(5, training, all_t)
conversion_rate = cluster_user(training, all_t)
print(conversion_rate)
