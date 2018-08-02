import numpy as np
import pandas as pd
from sklearn import metrics
import pickle
from sklearn.ensemble import IsolationForest


def isolation_forest(X, Y, contamination=0.1, n_estimators=50, bootstrap=True, validation=[]):

    if contamination == 'auto':
        contamination = Y.mean()
        print('Contamination Automatized to: %.2f\n' % contamination)

    db = IsolationForest(n_estimators=n_estimators, max_samples=X.shape[0],
                         bootstrap=bootstrap, verbose=1, random_state=42,
                         contamination=contamination)
    db.fit(X)

    labels = db.predict(X)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print('CLUSTER NUMBERS ', n_clusters_)
    print(labels)
    labels = pd.DataFrame(labels, columns=['outliers'])
    labels.loc[labels['outliers'] == 1, 'outliers'] = 0
    labels.loc[labels['outliers'] == -1, 'outliers'] = 1

    print('PRECISION %.4f' % metrics.precision_score(Y.values, labels.values))
    print('RECALL %.4f' % metrics.recall_score(Y.values, labels.values))
    print('F1 SCORE %.4f' % metrics.f1_score(Y.values, labels.values))

    if validation:
        assert validation[0].shape[1] > validation[1].shape[1], 'X valid has less columns than Y valid'
        predict_valid = db.predict(validation[0])
        predict_valid = pd.DataFrame(predict_valid, columns=['outliers'])
        predict_valid.loc[predict_valid['outliers'] == 1, 'outliers'] = 0
        predict_valid.loc[predict_valid['outliers'] == -1, 'outliers'] = 1

        print('PRECISION VALID %.4f' % metrics.precision_score(validation[1].values, predict_valid.values))
        print('RECALL VALID %.4f' % metrics.recall_score(validation[1].values, predict_valid.values))
        print('F1 SCORE VALID %.4f' % metrics.f1_score(validation[1].values, predict_valid.values))
    
    return labels


if __name__ == '__main__':
    import os

    seed = 42
    np.random.seed(seed)

    os.chdir("U:\\5FINDER\\resources\\test_data")

    pickle_off = open("Train.pickle", "rb")
    x = pickle.load(pickle_off)
    pickle_off = open("Valid.pickle", "rb")
    valid = pickle.load(pickle_off)
    '''
    json_off = open('Test.txt', 'r', encoding='utf8')
    x = json.load(json_off)
    '''
    columns = x[0]
    df = pd.DataFrame(x[1], columns=columns)
    df['p_churn'] = df['p_churn'].map(float)
    y = df['p_churn']
    del df['p_churn']
    del df['p_id']
    for i in df.columns.values.tolist():
        df[i] = df[i].map(float)
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    x = df


    valid = pd.DataFrame(valid[1], columns=columns)
    valid['p_churn'] = valid['p_churn'].map(float)
    y_valid = valid[['p_churn']]
    del valid['p_churn']
    del valid['p_id']
    for i in valid.columns.values.tolist():
        valid[i] = valid[i].map(float)
        valid[i] = (valid[i] - valid[i].mean()) / valid[i].std()

    isolation_forest(x, y, contamination='auto', validation=[valid, y_valid])