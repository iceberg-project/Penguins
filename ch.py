import numpy as np
def gen_folds(NIM,n):
    idx = np.random.permutation(NIM)
    return [idx[i::n] for i in xrange(n)]
print gen_folds(10,10)
print gen_folds(12,4)
print gen_folds(20,4)
print gen_folds(12,4)
print gen_folds(12,4)
print gen_folds(12,4)





