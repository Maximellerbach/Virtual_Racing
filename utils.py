import numpy as np

def transform_st(st, sq, mult):
    if st >= 0:
        return (st*mult)**sq
    return -np.absolute(st*mult)**sq

def smoothing_st(st, previous_st, delta_steer):
    # delta = st-previous_st
    # if delta>delta_steer:
    #     return previous_st+delta_steer
    # elif delta<-delta_steer:
    #     return previous_st-delta_steer
    # return st

    return np.average([st, previous_st])

def cat2linear(cat, coef=[-1, -0.5, 0, 0.5, 1], av=False):

    if av == True:
        st = 0
        count = 0
        for k in cat[0]:
            if k == True:
                count += 1

        if count != 0:
            for ait, nyx in enumerate(cat[0]):
                st += nyx*coef[ait]
            st/= count
        
    else:
        st = 0
        for ait, nyx in enumerate(cat[0]):
            st += nyx*coef[ait]
    return st
