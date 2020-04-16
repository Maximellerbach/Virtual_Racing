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
from utils import cat2linear, smoothing_st, transform_st, opt_acc, add_softmax, add_random
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
    def __init__(self, address, model, sleep_time=0.05, PID_settings=(0.5, 0.5, 1, 1), buffer_time=0.1, name='0', dir_save="C:\\Users\\maxim\\virtual_imgs\\"):
        super().__init__(*address, poll_socket_sleep_time=sleep_time)
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
        try:
            msg_type = json_packet['msg_type']
            if msg_type == "car_loaded":
                self.car_loaded = True
            
            elif msg_type == "telemetry":
                self.delay_buffer(json_packet, self.buffer_time)
                # self.last_image = image
                # self.to_process = True
                
                ### Disabled buffer for the moment
                imgString = json_packet["image"]
                tmp_img = np.asarray(Image.open(BytesIO(base64.b64decode(imgString))))
                self.last_image = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
                self.to_process = True
                self.current_speed = json_packet["speed"]

                # self.cte_history.append(json_packet["cte"])

            elif msg_type == "aborted":
                self.aborted = True

        except:
            pass
            # print(json_packet)

    def reset_car(self):
        msg = '{ "msg_type" : "reset_car" }'
        self.send(msg)

    def start(self, load_map=True, map_msg='', custom_body=True, body_msg='', custom_cam=True, cam_msg='', color=[20, 20, 20]):
        if load_map:
            if map_msg == '':
                msg = '{ "msg_type" : "load_scene", "scene_name" : "generated_track" }'
            else:
                msg = map_msg
            self.send(msg)
            # print("sended map!")

            loaded = False
            while(not loaded):
                time.sleep(1.0)
                loaded = self.car_loaded

        if custom_body:
            if body_msg == '':
                msg = '{ "msg_type" : "car_config", "body_style" : "car02", "body_r" : "'+str(color[0])+'", "body_g" : "'+str(color[1])+'", "body_b" : "'+str(color[2])+'", "car_name" : "RedPlex_'+self.name+'", "font_size" : "50" }'
            else:
                msg = body_msg
            self.send(msg)
            time.sleep(1.0)
            # print("sended custom body style!")

        if custom_cam:
            if cam_msg == '':
                msg = '{ "msg_type" : "cam_config", "fov" : "70", "fish_eye_x" : "0.0", "fish_eye_y" : "0.0", "img_w" : "255", "img_h" : "255", "img_d" : "3", "img_enc" : "JPG", "offset_x" : "0.0", "offset_y" : "1.7", "offset_z" : "1.0", "rot_x" : "40.0" }'
            else:
                msg = cam_msg
            self.send(msg)
            time.sleep(1.0)
            # print("sended custom camera settings!")

        # self.update(0, throttle=0, brake=0.1)
        # print("car loaded, ready to go!", self.name)

    def send_controls(self, steering, throttle, brake):
        p = { "msg_type" : "control",
                "steering" : steering.__str__(),
                "throttle" : throttle.__str__(),
                "brake" : brake.__str__()}
        msg = json.dumps(p)
        self.send(msg)

        #this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)


    def delay_buffer(self, img, buffer_time=0.1): # TODO: redo this function
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
                st = add_random(st)   

            self.update(st, throttle=optimal_acc)

            # cv2.imshow('img_'+str(self.name), img) # cause problems with threaded prediction
            # cv2.waitKey(1)

            self.to_process = False
            self.previous_st = st
            return st
        return 0


    def get_keyboard(self, disable_keys=["a"], keys=["left", "up", "right"], bkeys=["down"]):
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
        
        for dk in disable_keys:
            dpressed.append(keyboard.is_pressed(dk))

        if any(pressed+bpressed) and not any(dpressed):
            manual = True

        # print(keys_st, pressed)
        return manual, keys_st, bfactor

    def generate_random_color(self):
        return np.random.randint(0, 255, size=(3))
    
    def rdm_color_start(self, load_map=True, custom_body=True, custom_cam=False, color=[]):
        if color == []:
            color = self.generate_random_color()
        self.start(load_map=load_map, custom_body=custom_body, custom_cam=custom_cam, color=color)

    def save_img(self, img, st):
        direction = round(st*4)+7
        # print(direction) # should be [3, 5, 7, 9, 11]
        tmp_img = img[self.crop:]
        tmp_img = cv2.resize(tmp_img, (160,120))
        cv2.imwrite(self.dir_save+str(direction)+'_'+str(time.time())+'.png', tmp_img)
        

class auto_client(SimpleClient):
    def __init__(self, window, model, host = "127.0.0.1", port = 9091, sleep_time=0.05, PID_settings=(1, 10, 1, 0.5, 1, 1), buffer_time=0.1, name="0"):
        super().__init__((host, port), model, sleep_time=sleep_time, PID_settings=PID_settings, name=name, buffer_time=buffer_time)
        self.model._make_predict_function()
        self.rdm_color_start()
        self.loop_settings = [True, False, False]

        self.t = threading.Thread(target=self.loop)
        self.t.start()
        
        AutoInterface(window, self)

    def loop(self, cat2st=True, transform=True, smooth=True, random=False):
        self.update(0, throttle=0.0, brake=0.1)
        while(True):
            transform, smooth, random = self.loop_settings
            
            self.predict_st(cat2st=cat2st, transform=transform, smooth=smooth, random=random)

            if self.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.send(msg)
                time.sleep(1.0)

                self.stop()
                print("stopped client", self.name)
                break

class semiauto_client(SimpleClient):
    def __init__(self, window, model, host = "127.0.0.1", port = 9091, sleep_time=0.05, PID_settings=(1, 10, 1, 0.5, 1, 1), buffer_time=0.1, name="0"):
        super().__init__((host, port), model, sleep_time=sleep_time, PID_settings=PID_settings, name=name, buffer_time=buffer_time)
        self.model._make_predict_function()
        self.rdm_color_start()
        self.loop_settings = [True, False, False]

        self.t = threading.Thread(target=self.loop)
        self.t.start()
        
        AutoInterface(window, self)

    def loop(self, cat2st=True, record=False):
        self.update(0, throttle=0.0, brake=0.1)
        while(True):
            delta_steer, target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings
            transform, smooth, random = self.loop_settings

            toogle_manual, manual_st, bk = self.get_keyboard()

            if toogle_manual == True:
                throttle = opt_acc(manual_st, self.current_speed, max_throttle, min_throttle, target_speed)
                self.update(manual_st, throttle=throttle*bk)

                if record == True and self.to_process == True:
                    self.save_img(self.last_image, manual_st)

                if random:
                    manual_st = add_random(manual_st)
                self.to_process = False

            else:
                self.predict_st(cat2st=cat2st, transform=transform, smooth=smooth, random=random)
            
            if self.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.send(msg)
                time.sleep(1.0)

                self.stop()
                print("stopped client", self.name)
                break


class manual_client(SimpleClient):
    def __init__(self, window, model, host = "127.0.0.1", port = 9091, sleep_time=0.05, PID_settings=(1, 10, 1, 0.5, 1, 1), buffer_time=0.1, name="0"):
        super().__init__((host, port), "no model required here", sleep_time=sleep_time, PID_settings=PID_settings, name=name, buffer_time=buffer_time)
        self.rdm_color_start()
        self.loop_settings = [True, False, False]

        self.t = threading.Thread(target=self.loop)
        self.t.start()

        AutoInterface(window, self)

    def loop(self, cat2st=True, record=False):
        self.update(0, throttle=0.0, brake=0.1)
        while(True):
            delta_steer, target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings
            transform, smooth, random = self.loop_settings

            toogle_manual, manual_st, bk = self.get_keyboard()
            
            if toogle_manual == True:
                throttle = opt_acc(manual_st, self.current_speed, max_throttle, min_throttle, target_speed)

                if record == True and self.to_process==True:
                    self.save_img(self.last_image, manual_st)

                if random:
                    manual_st = add_random(manual_st, 0.3, 0.4)

                self.update(manual_st, throttle=throttle*bk)
                self.to_process = False

            else:
                self.update(0, throttle=0.0, brake=0.1)

            
            if self.aborted == True:
                msg = '{ "msg_type" : "exit_scene" }'
                self.send(msg)
                time.sleep(1.0)

                self.stop()
                print("stopped client", self.name)
                break


def select_mode(mode_index):
    obj_list = [auto_client, semiauto_client, manual_client]
    return obj_list[mode_index]


if __name__ == "__main__":
    model = load_model('C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\convolution\\lightv6_mix.h5', compile=False)
    
    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #     model = load_model('lane_keeper.h5')
    # model.summary()

    host = "127.0.0.1" # "127.0.0.1" # "trainmydonkey.com" for virtual racing server
    port = 9091
    interval= 1.0
    fps = 20
    sleep_time = 1/fps

    delta_steer = 0.05 # steering value where you consider the car to go straight
    target_speed = 10
    max_throttle = 1.0 # if you set max_throttle=min_throttle then throttle will be cte
    min_throttle = 0.5
    sq = 1 # modify steering by : st ** sq
    mult = 1 # modify steering by: st * mult
    buffer_time = 0.0

    clients_modes = [0, 0] # clients to spawn
    settings = [delta_steer, target_speed, max_throttle, min_throttle, sq, mult] # can be changed in graphic interface

    window = windowInterface() # create window

    ths = []
    load_map = True
    for i in range(len(clients_modes)):

        ### THREAD VERSION ### 
        client = select_mode(clients_modes[i])
        client(window, model, host=host, port=port, sleep_time=sleep_time, PID_settings=settings, buffer_time=buffer_time, name=str(i))
        print("started client", i)

        time.sleep(interval) # wait a bit to add an other client
        load_map = False

    window.mainloop() # only display when everything is loaded to avoid threads to be blocked !
