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
from keras.initializers import glorot_uniform
from keras.models import load_model
from keras.utils import CustomObjectScope
from PIL import Image

from gym_donkeycar.core.sim_client import SDClient
from utils import cat2linear, smoothing_st, transform_st, opt_acc

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # force tensorflow/keras to use the cpu instead of gpu (already used by the game)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


###########################################

class SimpleClient(SDClient):
    def __init__(self, address, model, PID_settings=(0.5, 0.5, 0.1, 1, 1), poll_socket_sleep_time=0.01, buffer_time=0.1, name='0', dir_save="C:\\Users\\maxim\\virtual_imgs\\"):
        super().__init__(*address, poll_socket_sleep_time=poll_socket_sleep_time)
        self.last_image = None
        self.car_loaded = False
        self.to_process = False
        self.aborted = False
        self.crop = 40
        self.previous_st = 0
        self.current_speed = 0
        self.packet_buffer = []
        self.cte_history = []

        self.name = name
        self.model = model
        self.PID_settings = PID_settings
        self.buffer_time = buffer_time
        self.dir_save = dir_save

    def on_msg_recv(self, json_packet):
        msg_type = json_packet['msg_type']
        if msg_type == "car_loaded":
            self.car_loaded = True
        
        elif msg_type == "telemetry":
            # self.delay_buffer(json_packet, self.buffer_time)
            imgString = json_packet["image"]
            tmp_img = np.asarray(Image.open(BytesIO(base64.b64decode(imgString))))
            self.last_image = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
            self.to_process = True
            
            # self.cte_history.append(json_packet["cte"])

        elif msg_type == "aborted":
            print(json_packet)
            self.aborted = True


    def start(self, load_map=True, map_msg='', custom_body=True, body_msg='', custom_cam=True, cam_msg='', color=[20, 20, 20]):
        if load_map:
            if map_msg == '':
                msg = '{ "msg_type" : "load_scene", "scene_name" : "generated_track" }'
            else:
                msg = map_msg
            self.send(msg)

        loaded = False
        while(not loaded):
            time.sleep(1.0)
            loaded = self.car_loaded
        print("sended map!")
        
        if custom_body:
            if body_msg == '':
                msg = '{ "msg_type" : "car_config", "body_style" : "car02", "body_r" : "'+str(color[0])+'", "body_g" : "'+str(color[1])+'", "body_b" : "'+str(color[2])+'", "car_name" : "RedPlex_'+self.name+'", "font_size" : "50" }'
            else:
                msg = body_msg
            self.send(msg)
        time.sleep(1)
        print("sended custom body style!")

        if custom_cam:
            if cam_msg == '':
                msg = '{ "msg_type" : "cam_config", "fov" : "70", "fish_eye_x" : "0.0", "fish_eye_y" : "0.0", "img_w" : "255", "img_h" : "255", "img_d" : "3", "img_enc" : "JPG", "offset_x" : "0.0", "offset_y" : "1.7", "offset_z" : "1.0", "rot_x" : "40.0" }'
            else:
                msg = cam_msg
            self.send(msg)

        time.sleep(1)
        print("sended custom camera settings!")

        print("car loaded, ready to go!")

    def send_controls(self, steering, throttle, brake):
        p = { "msg_type" : "control",
                "steering" : steering.__str__(),
                "throttle" : throttle.__str__(),
                "brake" : brake.__str__()}
        msg = json.dumps(p)
        self.send(msg)

        #this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)


    def delay_buffer(self, img, buffer_time=0.1): # TODO
        return

    def update(self, st, throttle=1.0, brake=0.0):
        self.send_controls(st, throttle, brake)

    def predict_st(self, cat2st=True, transform=True, smooth=True, random=True, coef=[-1, -0.5, 0, 0.5, 1]):
        if self.to_process==True:
            img = self.last_image
            delta_steer, target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings
            
            img = img[self.crop:, :, :] # crop img to be as close as training data as possible
            img = cv2.resize(img, (160,120))

            pred_img = np.expand_dims(img, axis=0)/255
            pred_img = pred_img+np.random.random(size=(1,120,160,3))/5 # add some noise to pred to test robustness

            ny = self.model.predict(pred_img)

            if cat2st: 
                st = cat2linear(ny, coef=coef)

                if transform:
                    st = transform_st(st, sq, mult)
                if smooth:
                    st = smoothing_st(st, self.previous_st, delta_steer)

                
                optimal_acc = opt_acc(st, self.current_speed, max_throttle, min_throttle, target_speed)

            else:
                st, optimal_acc = ny
                st = st[0][0]
                optimal_acc = optimal_acc[0][0]
                
                if transform:
                    st = transform_st(st, sq, mult)
                if smooth:
                    st = smoothing_st(st, self.previous_st, delta_steer)

            if random:
                random_dir = np.random.choice([True, False], p=[0.5, 0.5]) # add some noise to the direction to see robustness
                if random_dir:
                    st = st+np.random.random()/2.5
            
            self.update(st, throttle=optimal_acc)

            # cv2.imshow('img_'+str(self.name), img) # cause problems with threaded prediction
            # cv2.waitKey(1)

            self.to_process = False
            self.previous_st = st

    def save_img(self, img, st): # TODO: do this function
        direction = round(st*4)+7
        # print(direction) # should be [3, 5, 7, 9, 11]
        tmp_img = img[self.crop:]
        cv2.resize(tmp_img, (160,120))
        cv2.imwrite(self.dir_save+str(direction)+'_'+str(time.time())+'.png', tmp_img)

    def get_keyboard(self, keys=["left", "up", "right"], backward_k=["down"]): # TODO: do this function
        pressed = []
        bpressed = []
        manual = False
        backward_factor = 1

        for k in keys:
            pressed.append(keyboard.is_pressed(k))
        keys_st = cat2linear([pressed], coef=[-1, 0, 1], av=True)
        
        for bk in backward_k:
            bpressed.append(keyboard.is_pressed(bk))

        if any(bpressed):
            backward_factor = -1

        if any(pressed+bpressed):
            manual = True


        # print(keys_st, pressed)
        return manual, keys_st, backward_factor


class predicting_client():
    def __init__(self, model, host = "127.0.0.1", port = 9091, PID_settings=(1, 10, 1, 1, 1), buffer_time=0.1, name="0"):
        self.host = host
        self.port = port
        self.PID_settings = PID_settings
        self.buffer_time = buffer_time
        self.name = name
        self.client = SimpleClient((self.host, self.port), model, PID_settings=self.PID_settings, name=self.name, buffer_time=self.buffer_time)

    def start(self, load_map=True, custom_body=True, custom_cam=False, color=[]):
        if color == []:
            color = self.generate_random_color()
        self.client.start(load_map=load_map, custom_body=custom_body, custom_cam=custom_cam, color=color)

    def generate_random_color(self):
        return np.random.randint(0, 255, size=(3))

    def autonomous_loop(self, load_map=True, cat2st=True, transform=True, smooth=True, random=False, record=False): # TODO: add record on autonomous (pass by predict_st for last_img record)
        self.start(load_map=load_map, custom_body=True, custom_cam=False)
        while(True):
            self.client.predict_st(cat2st=cat2st, transform=transform, smooth=smooth, random=random)

            if self.client.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.client.send(msg)
                time.sleep(1.0)

                self.client.stop()
                print("stopped client", self.name)
                break

    def autonomous_manual_loop(self, load_map=True, cat2st=True, transform=True, smooth=True, random=False, record=False):
        self.start(load_map=load_map, custom_body=True, custom_cam=False)
        delta_steer, target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings

        while(True):
            toogle_manual, manual_st, bk = self.client.get_keyboard()
            
            if toogle_manual == True:
                if record == True and self.client.to_process==True:
                    self.client.save_img(self.client.last_image, manual_st)

                optimal_acc = opt_acc(manual_st, self.client.current_speed, max_throttle, min_throttle, target_speed)
                self.client.update(manual_st, throttle=optimal_acc*bk)
                self.client.to_process = False

            else:
                self.client.predict_st(cat2st=cat2st, transform=transform, smooth=smooth, random=random)
            
            if self.client.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.client.send(msg)
                time.sleep(1.0)

                self.client.stop()
                print("stopped client", self.name)
                break

    def manual_loop(self, load_map=True, cat2st=True, transform=True, smooth=True, random=False, record=False):
        self.start(load_map=load_map, custom_body=True, custom_cam=False)
        delta_steer, target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings

        while(True):
            toogle_manual, manual_st, bk = self.client.get_keyboard()
            if toogle_manual == True:    
                if record == True and self.client.to_process==True:
                    self.client.save_img(self.client.last_image, manual_st)
                
                optimal_acc = opt_acc(manual_st, self.client.current_speed, max_throttle, min_throttle, target_speed)
                self.client.update(manual_st, throttle=optimal_acc*bk)
                self.to_process = False

            else:
                self.client.update(manual_st, throttle=0.0, brake=0.1)

            if self.client.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.client.send(msg)
                time.sleep(1.0)

                self.client.stop()
                print("stopped client", self.name)
                break

    def mode_list(self, mode_index):
        modes = [self.autonomous_loop, self.autonomous_manual_loop, self.manual_loop]
        return modes[mode_index]

if __name__ == "__main__":
    model = load_model('C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\convolution\\lightv6_mix.h5', compile=False)
    
    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #     model = load_model('lane_keeper.h5')
    # model.summary()

    host = "127.0.0.1" # "trainmydonkey.com" for virtual racing server
    port = 9091
    interval= 5.0
    fps = 20

    delta_steer = 0.05 # do not needed if np.average is used in smoothing function
    target_speed = 10
    max_throttle = 0.8 # setting max_throttle = min_throttle will get you a constant speed
    min_throttle = 0.2
    sq = 0.75 # modify steering by: s**sq ; depends on the sensivity you want
    mult = 1. # modify steering by: s*mult ; usually 1
    buffer_time = 0.0

    settings = (delta_steer, target_speed, max_throttle, min_throttle, sq, mult)
    client_mode = [0, 0, 2] # select here clients mode
    client_settings = [settings]*len(client_mode) # basic list to make every client have the same settings
    clients = []
    Threads = []

    load_map = False
    for i, mode in enumerate(client_mode):
        driving_client = predicting_client(model, host=host, port=port, PID_settings=client_settings[i], buffer_time=buffer_time, name=str(i)) # PID_settings=settings)
        
        ### THREAD VERSION ### 
        driving_client.client.model._make_predict_function()
        t = threading.Thread(target=driving_client.mode_list(mode), args=(load_map, True, True, True, False, False))
        t.start()
        Threads.append(t)
        print("started Thread", i)


        ### NORMAL VERSION ### 
        # clients[i].autonomous_loop(load_map=load_map, cat2st=True, transform=True, smooth=True, random=False, record=False)

        ### MANUAL RECOVERY VERSION ###
        # clients[i].autonomous_manual_loop(cat2st=True, transform=True, smooth=True, random=False, record=False)
        
        ### MANUAL VERSION ###
        # clients[i].manual_loop(cat2st=True, transform=True, smooth=False, random=False, record=False)

        time.sleep(interval)
