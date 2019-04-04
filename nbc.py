import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
df = pd.read_csv('./SpamInstances.txt', delimiter=' ', header=0, names=['number', 'label', 'vector'])
spam_vectors = df.loc[df['label'] == 1]
nonspam_vectors = df.loc[df['label'] == -1]
print('spam\n', spam_vectors)
print('non spam\n', nonspam_vectors)

datas = np.array(df)
labels = datas[:,1]
feature_vectors = datas[:,2]

# def bayes(vectors):
    # denominator remains the same for the whole calc
    # top of the fraction is the product of
    # for i in range(len(vectors[0])):

def partition():
    for i in range(155):
        num_instances = i * 100 + 100
        train = df[df['label'] == 1].sample(n=int(num_instances*0.4)) + df[df['label'] == -1].sample(n=int(num_instances*0.40))
        test = df[df['label'] == 1].sample(n=int(num_instances*0.10)) + df[df['label'] == -1].sample(n=int(num_instances*0.10))


# partition()
def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return accuracy_score(y, self.predict(X))


clf = BernoulliNB(binarize=0.0)
num_instances = 2000
train = spam_vectors.sample(n=int(num_instances*0.4)).append(nonspam_vectors.sample(n=int(num_instances*0.40)))
test = shuffle(spam_vectors.sample(n=int(num_instances*0.1)).append(nonspam_vectors.sample(n=int(num_instances*0.10))))
print(train)
print(train['label'])
vectors = np.array(train['vector'])

binary_vectors =  [[int(c) for c in v] for v in vectors]
clf = clf.fit(binary_vectors, np.array(train['label']))

print(clf.predict(binary_vectors[2:3]), np.array(train['label'])[2])

def binary_vectors(vectors):
    return np.array([[c for c in v] for v in vectors]).astype(np.float64)

test_bin = binary_vectors(np.array(test['vector']))

print(clf.predict(test_bin[:10]), np.array(test['label'])[:10])

print(score(clf, test_bin, np.array(test['label'])))

def binarize_vector(vector):
    return map(conv, vector)

def conv(i):
    if i == -1:
        return 0
    else:
        return 1
