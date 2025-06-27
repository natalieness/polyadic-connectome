''' From maggot_models, based on signal flow metric from Varshney et al 2011'''

import numpy as np
from graspologic.utils import remove_loops


def signal_flow(A):

    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z