import random
from typing import List, Dict, Tuple, Callable
import numpy as np
import scipy.stats

def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

def create_folds(xs: List, n: int) -> List[List[List]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def create_train_test(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test

def euclidean_dist(x1,x2,w=1):
    # Calculate the distance between two points
    x1 = np.array(x1)
    x2 = np.array(x2)
    diffsq = (x1 - x2)**2
    return np.sqrt(np.sum(w*diffsq))

def split_feat_label(x,idx):
    # Split data at idx for label and features. Input is a list.
    x = list(x)
    return x[idx], x[:idx] + x[min(idx+1,len(x)):]

def sort_dist(dist_list):
    return sorted(dist_list,key=lambda x: x['d'], reverse=False)

def keep_nearest_neighbors(dist, dist_list):
    k = len(dist_list)
    dist_list = sort_dist(dist_list)
    for i,el in enumerate(dist_list):
        if dist['d'] < el['d']:
            return dist_list[:i] + [dist] + dist_list[min(i,k-1):k-1]
    return dist_list

def build_knn(database,idx_label=0):
    def knn(k,query):
        q_label, q_feat = split_feat_label(query, idx_label)
        nearest_neighbors = []
        for p in database:
            p_label, p_feat = split_feat_label(p, idx_label)
            dqp = euclidean_dist(q_feat, p_feat)
            dist = {'d':dqp,'y':p_label,'x':p_feat}
            if len(nearest_neighbors) < k:
                nearest_neighbors += [dist]
            else:
                nearest_neighbors = keep_nearest_neighbors(dist, nearest_neighbors)
                
        dist_nn_sum = np.sum([nn['d'] for nn in nearest_neighbors])
        if dist_nn_sum <= 0:
            y_pred = np.mean([nn['y'] for nn in nearest_neighbors])
            y_err = np.sqrt(np.sum([(nn['y'] - y_pred)**2 for nn in nearest_neighbors]))
        else:
            y_pred = np.sum([nn['d']*nn['y'] for nn in nearest_neighbors])/dist_nn_sum
            y_err = np.sqrt(np.sum([nn['d']*(nn['y'] - y_pred)**2 for nn in nearest_neighbors])/dist_nn_sum)

        return {'y_pred':y_pred,'y_err':y_err,'nearest_neighbors':nearest_neighbors}
    return knn

def mean_confidence_interval(data: List[float], confidence: float=0.95) -> Tuple[float, Tuple[float, float]]:
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, (m-h, m+h)

def extract_minmax(data):
    # Extract range of values - brute force method
    ndim = len(data[0])
    xmin, xmax = data[0].copy(), data[0].copy()
    for i in range(ndim):
        for p in data:
            xmin[i] = p[i] if p[i] < xmin[i] else xmin[i]
            xmax[i] = p[i] if p[i] > xmax[i] else xmax[i]

    return [(xmin[i], xmax[i]) for i in range(ndim)]

if __name__ == '__main__':
    data = parse_data("concrete_compressive_strength.csv")
    folds = create_folds(data, 10)
    train, test = create_train_test(folds, 0)
    knn = build_knn(train,idx_label=8)
    for i in range(3):
        result = knn(10, test[i])
        y_pred = result['y_pred']
        y_err = result['y_err']
        print(f"Test value: {test[i][-1]:.2f} Predicted value: {y_pred:.3f} +/- {y_err:.3f}")