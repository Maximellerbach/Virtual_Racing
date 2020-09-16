from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
import numpy as np
import time


def safe_load_model(path, compile=True):
    try:
        return load_model(path, compile=compile)
    except ValueError:
        with tfmot.sparsity.keras.prune_scope():
            return load_model(path, compile=compile)


def get_model_output_names(model):
    tot_out_names = ['left_lane', 'right_lane', 'direction', 'throttle']
    output_names = [node.op.name.split('/')[0] for node in model.outputs]
    return [name for name in tot_out_names for output_name in output_names if name in output_name]


def prediction2dict(predictions, model_output_names):
    predictions_list = [[]]*len(predictions[0])
    for prediction in predictions:
        for pred_number, pred in enumerate(prediction):
            predictions_list[pred_number].append(pred)

    output_dicts = [{output_name: [] for output_name in model_output_names}
                    for _ in range(len(predictions_list))]
    for prediction, output_dict in zip(predictions_list, output_dicts):
        for output_value, output_name in zip(prediction, output_dict):
            output_dict[output_name] = output_value[0]
    return output_dicts


def predict_decorator(func, model_output_names):
    def wrapped_f(*args, **kwargs):
        st = time.time()
        predictions = func(*args, **kwargs)
        output_dicts = prediction2dict(predictions, model_output_names)
        et = time.time()
        return (output_dicts, et-st)
    return wrapped_f


def apply_predict_decorator(model):
    model.predict = predict_decorator(
        model.predict, get_model_output_names(model))
