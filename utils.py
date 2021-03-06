import numpy as np
from keras.models import Model, Input
from keras.layers import Activation

def transform_st(st, sq, mult):
    if st >= 0:
        return (st*mult)**sq
    return -np.absolute(st*mult)**sq

def smoothing_st(st, previous_st, thr_steer):
    if np.absolute(st)>=thr_steer:
        return np.average([st, previous_st])
    return 0

def opt_acc(st, current_speed, max_throttle, min_throttle, target_speed):
    dt_throttle = max_throttle-min_throttle

    optimal_acc = ((target_speed-current_speed)/target_speed)
    if optimal_acc < 0:
        optimal_acc = 0

    optimal_acc = min_throttle+((optimal_acc**0.1)*(1-np.absolute(st)))*dt_throttle

    if optimal_acc > max_throttle:
        optimal_acc = max_throttle
    elif optimal_acc < min_throttle:
        optimal_acc = min_throttle

    return optimal_acc

def boost(st):
    return

def add_random(st, frc=0.5, mult=0.2):
    rdm_bool = np.random.choice([True, False], p=[frc, 1-frc]) # add some noise to the direction to see robustness
    if rdm_bool:
        return st+(np.random.random()-0.5)*mult
    return st
    
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

def st2cat(st): # maps every -1; 1 value to rounded direction value : [3, 5, 7, 9, 11]
    return int(round(st*2)*2+7)

def add_softmax(model, shape=(120, 160, 3)): # TODO
    inp = Input(shape)

    x = model(inp)
    y = Activation('softmax')(x)

    new_model = Model(inp, y)
    return new_model

# TODO: function to calculate optimal st with current speed (not needed for the moment)