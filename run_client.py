import base64
import json
import threading
import time
from io import BytesIO

import cv2
import keyboard
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from PIL import Image
from custom_modules.datasets import dataset_json

from gym_donkeycar.core.sim_client import SDClient
from utils import cat2linear, transform_st, opt_acc, add_random
from visual_interface import windowInterface, AutoInterface


physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)


class SimpleClient(SDClient):
    def __init__(self, address, model, dataset, name='0',
                 PID_settings=(0.5, 0.5, 1, 1), buffer_time=0.1, use_speed=(False, False), sleep_time=0.01):
        super().__init__(*address, poll_socket_sleep_time=sleep_time)
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
        self.dataset = dataset
        self.show_img = True
        self.PID_settings = PID_settings
        self.buffer_time = buffer_time
        self.default_dos = f'C:\\Users\\maxim\\recorded_imgs\\{self.name}_{time.time()}\\'

    def on_msg_recv(self, json_packet):
        try:
            msg_type = json_packet['msg_type']

            if msg_type == "car_loaded":
                self.car_loaded = True

            elif msg_type == "aborted":
                self.aborted = True

            elif msg_type == "telemetry":
                imgString = json_packet["image"]
                tmp_img = np.asarray(Image.open(
                    BytesIO(base64.b64decode(imgString))))
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
                self.delay_buffer(tmp_img)

                if self.show_img:
                    cv2.imshow('img_'+self.name, tmp_img)
                    cv2.waitKey(1)

                self.to_process = True
                self.current_speed = json_packet["speed"]

                del json_packet["image"]
                # print(json_packet)

        except:
            if json_packet != {}:
                print(json_packet)
            pass

    def reset_car(self):
        msg = {"msg_type": "reset_car"}
        self.send_now(json.dumps(msg))

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

    def startv1(self, custom_body=True, body_msg='', custom_cam=False, cam_msg='', color=[20, 20, 20]):
        if custom_body:  # send custom body info
            if body_msg == '':
                msg = {
                    "msg_type": "car_config",
                    "body_style": "car02",
                    "body_r": str(color[0]),
                    "body_g": str(color[1]),
                    "body_b": str(color[2]),
                    "car_name": f'Maxime_{self.name}',
                    "font_size": "50"}
            else:
                msg = body_msg
            self.send(json.dumps(msg))
            time.sleep(1.0)

        if custom_cam:  # send custom cam info
            if cam_msg == '':
                msg = {
                    "msg_type": "cam_config",
                    "fov": "0",
                    "fish_eye_x": "0.0",
                    "fish_eye_y": "0.0",
                    "img_w": "255",
                    "img_h": "255",
                    "img_d": "3",
                    "img_enc": "JPG",
                    "offset_x": "0.0",
                    "offset_y": "1.7", "offset_z":
                    "1.0", "rot_x": "40.0"
                }
            else:
                msg = cam_msg
            self.send(json.dumps(msg))
            time.sleep(1.0)

    def startv2(self, name='Maxime', cam_msg='', color=[20, 20, 20]):
        '''
        send three config messages to setup car, racer, and camera
        '''
        racer_name = "Maxime Ellerbach"
        car_name = name+'_'+self.name
        bio = "I race robots."  # TODO
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
        msg = '{ "msg_type" : "car_config", "body_style" : "donkey", "body_r" : "'+str(color[0])+'", "body_g" : "'+str(
            color[1])+'", "body_b" : "'+str(color[2])+'", "car_name" : "%s", "font_size" : "100" }' % (car_name)
        self.send_now(msg)
        print("sended body info")

        # this sleep gives the car time to spawn. Once it's spawned, it's ready for the camera config.
        time.sleep(0.1)

        # using default cam
        # msg = '{ "msg_type" : "cam_config", "fov" : "70", "fish_eye_x" : "0.0", "fish_eye_y" : "0.0", "img_w" : "255", "img_h" : "255", "img_d" : "3", "img_enc" : "JPG", "offset_x" : "0.0", "offset_y" : "1.7", "offset_z" : "1.0", "rot_x" : "40.0" }'
        # msg = '{ "msg_type" : "cam_config", "fov" : "150", "fish_eye_x" : "1.0", "fish_eye_y" : "1.0", "img_w" : "255", "img_h" : "255", "img_d" : "1", "img_enc" : "JPG", "offset_x" : "0.0", "offset_y" : "3.0", "offset_z" : "0.0", "rot_x" : "90.0" }'
        # self.send_now(msg)
        # print("sended cam info")

    def send_controls(self, steering, throttle, brake):
        p = {"msg_type": "control",
             "steering": steering.__str__(),
             "throttle": throttle.__str__(),
             "brake": brake.__str__()}

        self.send(json.dumps(p))

        # this sleep lets the SDClient thread poll our message and send it out.
        time.sleep(self.poll_socket_sleep_sec)

    def update(self, st, throttle=1.0, brake=0.0):
        self.send_controls(st, throttle, brake)

    def prepare_img(self, img):
        # shouldn't be necessary but may be usefull in the future
        img = img[self.crop:, :, :]
        img = cv2.resize(img, (160, 120))
        return img

    def predict_st(self, cat2st=False, transform=True, smooth=True, record=False, coef=[-1, -0.5, 0, 0.5, 1]):
        if self.to_process is False:
            return False

        target_speed, max_throttle, min_throttle, sq, mult = self.PID_settings

        img = self.prepare_img(self.last_image)
        pred_img = np.expand_dims(img, axis=0)/255

        if self.use_speed[0]:
            ny = self.model.predict(
                [pred_img, np.expand_dims(self.current_speed, axis=0)])
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
            optimal_acc = opt_acc(st, self.current_speed,
                                  max_throttle, min_throttle, target_speed)

        if transform:
            st = transform_st(st, sq, mult)

        if record:
            self.save_img(img, direction=st, speed=self.current_speed,
                          throttle=optimal_acc, time=time.time())

        self.update(st, throttle=optimal_acc, brake=0)
        self.to_process = False
        self.previous_st = st
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

    def rdm_color_startv1(self, color=[]):
        if color == []:
            color = np.random.randint(0, 255, size=(3))
        self.startv1(color=color)

    def rdm_color_startv2(self, color=[]):
        if color == []:
            color = np.random.randint(0, 255, size=(3))
        self.startv2(color=color)

    def save_img(self, img, direction=0, speed=None, throttle=None, time=None):
        tmp_img = self.prepare_img(img)

        to_save = {}
        if direction is not None:
            to_save['direction'] = direction
        if speed is not None:
            to_save['speed'] = speed
        if throttle is not None:
            to_save['throttle'] = throttle
        if time is not None:
            to_save['time'] = time

        self.dataset.save_img_and_annotation(
            self.default_dos, tmp_img, to_save)


class universal_client(SimpleClient):
    def __init__(self, params_dict, load_map):
        host = params_dict['host']
        port = params_dict['port']
        window = params_dict['window']
        use_speed = params_dict['use_speed']
        sleep_time = params_dict['sleep_time']
        PID_settings = params_dict['PID_settings']
        self.loop_settings = params_dict['loop_settings']
        self.record = params_dict['record']
        buffer_time = params_dict['buffer_time']
        name = params_dict['name']
        track = params_dict['track']
        model = params_dict['model']
        dataset = params_dict['dataset']

        super().__init__((host, port), model, dataset, sleep_time=sleep_time, name=name,
                         PID_settings=PID_settings, buffer_time=buffer_time, use_speed=use_speed)
        # self.model._make_predict_function() # useless with tf2
        
        if load_map:
            self.load_map(track)

        self.rdm_color_startv1()

        self.t = threading.Thread(target=self.loop)
        self.t.start()

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
                        throttle = opt_acc(
                            manual_st, self.current_speed, max_throttle, min_throttle, target_speed)
                else:
                    throttle = None

                if self.record and self.to_process:
                    self.save_img(self.last_image, direction=manual_st, speed=self.current_speed,
                                  throttle=throttle, time=time.time())
                    self.to_process = False

                if do_overide:
                    if random:
                        manual_st = add_random(manual_st, 0.3, 0.4)
                    self.update(manual_st, throttle=throttle*bk)

                else:
                    self.predict_st(transform=transform, smooth=smooth)

            else:
                self.predict_st(transform=transform, smooth=smooth)

            if self.aborted:
                msg = '{ "msg_type" : "exit_scene" }'
                self.send(msg)
                time.sleep(1.0)

                self.stop()
                print("stopped client", self.name)
                break


if __name__ == "__main__":
    model = load_model(
        'C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\models\\linearv4_latency.h5', compile=False)
    dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])

    remote = False
    host = 'trainmydonkey.com' if remote else '127.0.0.1'
    port = 9091

    window = windowInterface()  # create window
    sleep_time = 0.01

    config = {
        'host': host,
        'port': port,
        'window': window,
        'use_speed': (True, True),
        'sleep_time': sleep_time,
        'PID_settings': [17, 1.0, 0.45, 1.0, 1.0],
        'buffer_time': 0.0,
        'track': 'warehouse',
        'name': '0',
        'model': model,
        'dataset': dataset,
        'record': True,
        'loop_settings': [False, False, False, False]
    }

    load_map = True
    client_number = 2
    for i in range(client_number):
        universal_client(config, load_map)
        print("started client", i)
        load_map = False

    window.mainloop()  # only display when everything is loaded to avoid threads to be blocked !
