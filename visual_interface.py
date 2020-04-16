from tkinter import *
from run_client import auto_client
# TODO: interface to tweak client

class AutoInterface():
    def __init__(self, client_class, screen_size=(512, 512), name="Auto_0"):
        self.client_class = client_class

        self.screen_size = screen_size
        self.name = name
        self.scales = [] # list of scales in the interface
        self.values = [] # list of values of scales in interface

        self.create_interface()


    def create_interface(self, labels=["steer_threshold", "target_speed", "max_throttle", "min_throttle", "sq", "mult"], from_to=[(0.2, 0), (0, 30), (0, 1), (0, 1), (0.5, 2), (0.5, 2)]):
        self.window = Tk(self.name)
        # self.window.geometry(str(self.screen_size[0])+"x"+str(self.screen_size[1]))

        Button(self.window, text="Respawn", command=self.respawn).grid(row=0, column=0)
        Button(self.window, text="Reset to default", command=self.set_slider_value).grid(row=0, column=1)

        
        for it, label, scale_range in zip(range(len(labels)), labels, from_to):
            value = DoubleVar() # dummy value
            s = Scale(self.window, resolution=0.1, variable=value, command=self.get_slider_value, label=label, length=75, width=15, from_=scale_range[0], to=scale_range[1])
            # s.pack()
            s.grid(row=1, column=it)
            self.scales.append(s)
        
        self.set_slider_value() # set to default values
        self.get_slider_value() # get those values and store them into self.values

    def respawn(self):
        self.client_class.reset_car()

    def get_slider_value(self, v=0):
        values = []
        for scale in self.scales:
            values.append(scale.get())

        self.client_class.PID_settings = values

    def set_slider_value(self, values=(0.1, 10, 1, 0.5, 1, 1)): # can be used to set to default
        assert len(values) == len(self.scales)

        for value, scale in zip(values, self.scales):
            scale.set(value)

    def update(self):
        self.window.update()


if __name__ == "__main__":
    client = auto_client("some shit")
    auto = AutoInterface(client)