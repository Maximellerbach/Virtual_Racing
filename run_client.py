import base64
import json
import os
import random
import threading
import time
from io import BytesIO

import cv2
import keyboard
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from PIL import Image

from gym_donkeycar.core.sim_client import SDClient
from utils import cat2linear, st2cat, smoothing_st, transform_st, opt_acc, add_softmax, add_random
from visual_interface import windowInterface, AutoInterface

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # force tensorflow/keras to use the cpu instead of gpu (already used by the game)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


###########################################

class SimpleClient(SDClient):
    def __init__(self, address, model, Dataset, sleep_time=0.01, PID_settings=(0.5, 0.5, 1, 1), buffer_time=0.1, name='0', use_speed=(False, False)):
        super().__init__(*address, poll_socket_sleep_time=sleep_time)
        self.Dataset = Dataset
        self.last_image = np.zeros((120, 160, 3))
        self.car_loaded = False
        self.to_process = False
        self.aborted = False
        self.use_speed = use_speed
        self.crop = 40
        self.previous_st = 0
        self.current_speed = 0
        self.img_buffer = []
        self.cte_history = []
        self.previous_brake = []

        self.name = str(name)
        self.model = model
        self.PID_settings = PID_settings
        self.buffer_time = buffer_time

    def on_msg_recv(self, json_packet):
        try:
            msg_type = json_packet['msg_type']

            if msg_type == "need_car_config":
                self.rdm_color_startv2()

            elif msg_type == "car_loaded":
                self.car_loaded = True

            elif msg_type == "aborted":
                self.aborted = True

            elif msg_type == "telemetry":
                imgString = json_packet["image"]
                tmp_img = np.asarray(Image.open(BytesIO(base64.b64decode(imgString))))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
                self.delay_buffer(tmp_img)
                cv2.imshow('img_'+self.name, tmp_img)
                cv2.waitKey(1)

                self.to_process = True
                self.current_speed = json_packet["speed"]

                del json_packet["image"]
                # print(json_packet)

        except:
            print(json_packet)
            pass

    def reset_car(self):
        msg = '{ "msg_type" : "reset_car" }'
        self.send(msg)

    def delay_buffer(self, img):
        now = time.time()
        self.img_buffer.append((img, now))
        temp_img = None
        to_remove = []
        for it, imgt in enumerate(self.img_buffer):
            if now-imgt[1] > self.buffer_time:
                temp_img = imgt[0]
                to_remove.append(it)
            else:
                break
        if len(to_remove) > 0:
            self.last_image = temp_img
            for _ in range(len(to_remove)):
                del self.img_buffer[0]
                del to_remove[0]

    def load_map(self, track):
        msg = '{ "msg_type" : "load_scene", "scene_name" : "'+track+'" }'
        self.send_now(msg)

    def startv2(self, name='Maxime', cam_msg='', color=[20, 20, 20]):
        '''
        send three config messages to setup car, racer, and camera
        '''
        racer_name = "Maxime Ellerbach"
        car_name = name+'_'+self.name
        bio = "I race robots."
        country = "France"

        # Racer info
        msg = {'msg_type': 'racer_info',
               'racer_name': racer_name,
               'car_name': car_name,
               'bio': bio,
               'country': country}
        self.send_now(json.dumps(msg))
        print("sended racer info")

        # Car config
        msg = f'"msg_type": "car_config", "body_style": "donkey", "body_r": "{color[0]}", "body_g": "{color[1]}", "body_b": "{color[2]}", "car_name": "{car_name}", "font_size": "100"'
        self.send_now(msg)
        print("sended body info")

        # this sleep gives the car time to spawn. Once it's spawned, it's ready for the camera config.
        time.sleep(0.1)

    def send_controls(self, steering, throttle, brake):
        p = {"msg_type": "control",
             "steering": steering.__str__(),
             "throttle": throttle.__str__(),
             "brake": brake.__str__()}

        msg = json.dumps(p)
        self.send(msg)

        # this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)

    def update(self, st, throttle=1.0, brake=0.0):
        self.send_controls(st, throttle, brake)

    def prepare_img(self, img):
        # shouldn't be necessary but may be usefull in the future
        img = img[self.crop:, :, :]
        img = cv2.resize(img, (160, 120))
        return img

    def predict_st(self, cat2st=True, transform=True, smooth=True, record=False, coef=[-1, -0.5, 0, 0.5, 1]):
        if self.to_process is False:
            return False

        img = self.last_image
        target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings

        img = self.prepare_img(img)
        pred_img = np.expand_dims(img, axis=0)/255

        if self.use_speed[0]:
            ny = self.model.predict([pred_img, np.expand_dims(self.current_speed, axis=0)])
        else:
            ny = self.model.predict(pred_img)

        if cat2st:
            st = cat2linear(ny, coef=coef)
        elif self.use_speed[1]:
            st = ny[0][0][0]
        else:
            st = ny[0][0]

        if self.use_speed[1]:
            optimal_acc = ny[1][0][0]
            if optimal_acc > max_throttle:
                optimal_acc = max_throttle

        else:
            optimal_acc = opt_acc(st, self.current_speed, max_throttle, min_throttle, target_speed)

        if transform:
            st = transform_st(st, sq, mult)

        self.update(st, throttle=optimal_acc, brake=0)

        self.to_process = False
        self.previous_st = st

        if record:
            pass

        return st

    def get_keyboard(self, keys=["left", "up", "right"], bkeys=["down"]):
        pressed = []
        bpressed = []
        dpressed = []
        bfactor = 1
        manual = False

        for k in keys:
            pressed.append(keyboard.is_pressed(k))
        keys_st = cat2linear([pressed], coef=[-1, 0, 1], av=True)

        for bk in bkeys:
            bpressed.append(keyboard.is_pressed(bk))
        if any(bpressed):
            bfactor = -1

        if any(pressed+bpressed) and not any(dpressed):
            manual = True

        # print(keys_st, pressed)
        return manual, keys_st, bfactor

    def get_throttle(self, keys=["c", "d", "e"], bkeys=["down"]):
        pressed = []
        manual = False

        for k in keys:
            pressed.append(keyboard.is_pressed(k))
        keys_th = cat2linear([pressed], coef=[0, 0.5, 1], av=True)

        if any(pressed):
            manual = True

        return manual, keys_th

    def rdm_color_startv2(self, color=[]):
        if color == []:
            color = np.random.randint(0, 255, size=(3))
        self.startv2(color=color)

    def save_img(self, img, annotations={'direction': 0, 'speed': None, 'throttle': None}):
        # TODO: finish this function
        tmp_img = img[self.crop:]
        tmp_img = cv2.resize(tmp_img, (160, 120))

        direction = annotations['direction']
        speed = annotations['speed']
        throttle = annotations['throttle']

        to_save_annotations = [direction]

        # self.Dataset.save


class universal_client(SimpleClient):
    def __init__(self, params_dict):
        host = params_dict['host']
        port = params_dict['port']
        window = params_dict['window']
        use_speed = params_dict['use_speed']
        sleep_time = params_dict['sleep_time']
        PID_settings = params_dict['PID_settings']
        buffer_time = params_dict['buffer_time']
        name = params_dict['name']
        load_map = params_dict['load_map']
        track = params_dict['track']
        model = params_dict['model']
        self.save_path = params_dict['save_path']

        super().__init__((host, port), model, sleep_time=sleep_time, name=name,
                         PID_settings=PID_settings, buffer_time=buffer_time, use_speed=use_speed)
        self.model._make_predict_function()

        self.loop_settings = [True, False, False, False]
        self.record = False

        self.t = threading.Thread(target=self.loop())

        AutoInterface(window, self)

    def loop(self):
        self.update(0, throttle=0.0, brake=0.1)
        while(True):
            target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings
            transform, smooth, random, do_overide = self.loop_settings

            toogle_manual, manual_st, bk = self.get_keyboard()

            if toogle_manual:
                if transform:
                    manual_st = transform_st(manual_st, sq, mult)

                if self.use_speed[1]:
                    manual, throttle = self.get_throttle()
                    if manual is False:
                        throttle = opt_acc(manual_st, self.current_speed, max_throttle, min_throttle, target_speed)
                else:
                    throttle = None

                if self.record and self.to_process:
                    self.save_img(self.last_image, manual_st, self.save_path, speed=self.current_speed, throttle=throttle)
                    self.to_process = False

                if do_overide:
                    if random:
                        manual_st = add_random(manual_st, 0.3, 0.4)
                    self.update(manual_st, throttle=throttle*bk)

                else:
                    self.predict_st(transform=transform, smooth=smooth, random=random)
                

            else:
                self.predict_st(transform=transform, smooth=smooth, random=random)
            
            if self.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.send(msg)
                time.sleep(1.0)

                self.stop()
                print("stopped client", self.name)
                break


def select_mode(mode_index):
    obj_list = [auto_client, universal_client, manual_client]
    return obj_list[mode_index]


if __name__ == "__main__":
    # os.system("C:\\Users\\maxim\\GITHUB\\Virtual_Racing\\DonkeySimWin\\donkey_sime.exe") #start sim doesn't work for the moment
    model = load_model('C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\convolution\\linearv5_latency.h5', compile=False)

    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #     model = load_model('lane_keeper.h5')
    # model.summary()

    paths = "C:\\Users\\maxim\\recorded_imgs\\1_"+self.name+"_"+str(time.time())+"\\" # for the moment stays here, no need to specify the save path

    config = 1

    if config == 0:
        host = "trainmydonkey.com"
        sleep_time = 0.01
        target_speed = 17
        max_throttle = 1.0 # if you set max_throttle=min_throttle then throttle will be cte
        min_throttle = 0.45
        sq = 1.0 # modify steering by : st ** sq # can correct some label smoothing effects
        mult = 1.0 # modify steering by: st * mult (act kind as a sensivity setting)
        fake_delay = 0.0

    elif config == 1:
        host = "127.0.0.1"
        sleep_time = 0.01
        target_speed = 17
        max_throttle = 1.0 # if you set max_throttle=min_throttle then throttle will be cte
        min_throttle = 0.45
        sq = 1.0 # modify steering by : st ** sq # can correct some label smoothing effects
        mult = 1.0 # modify steering by: st * mult (act kind as a sensivity setting)
        fake_delay = 0.16


    clients_modes = [1] # clients to spawn : 0= auto; 1= semi-auto; 2= manual
    interval= 1.0
    port = 9091
    v1 = True

    settings = [target_speed, max_throttle, min_throttle, sq, mult] # can be changed in graphic interface
    window = windowInterface() # create window

    load_map = True
    for i in range(len(clients_modes)):

        ### THREAD VERSION ### 
        client = select_mode(clients_modes[i])
        client(window, model, host=host, port=port, cat2st=False, track='mountain_track', sleep_time=sleep_time, PID_settings=settings, name=str(i), thread=True, use_speed=(True, True), buffer_time=fake_delay, load_map=load_map, v1=v1)
        print("started client", i)

        # time.sleep(interval) # wait a bit to add an other client
        load_map = False

    window.mainloop() # only display when everything is loaded to avoid threads to be blocked !
