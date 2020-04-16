from tkinter import *

class windowInterface(Tk): # screen to display interface for ALL client
    def __init__(self, name="basic interface"):
        super().__init__(name)
        self.n_client = 0
        self.off_y = 0

    def increment(self, space=2):
        self.n_client += 1
        self.off_y += space

class AutoInterface(): # single interface object for 1 client
    def __init__(self, window, client_class, screen_size=(512, 512), name="Auto_0"):
        self.window = window
        self.client_class = client_class

        self.screen_size = screen_size
        self.name = name

        self.scales_value = [] # list of scales objects in interface
        self.values = [] # list of values of scales in interface
        self.bool_checkbox = [] # list of checkbox objects in interface

        self.add_interface()


    def add_interface(self, scale_labels=["steer_threshold", "target_speed", "max_throttle", "min_throttle", "sq", "mult"], from_to=[(0.2, 0), (0, 30), (0, 1), (0, 1), (0.5, 2), (0.5, 2)]):
        off_y = self.window.off_y
        Button(self.window, text="Respawn", command=self.respawn).grid(row=off_y, column=0)
        Button(self.window, text="Reset to default", command=self.set_slider_value).grid(row=off_y, column=1)


        bvar = BooleanVar()
        b = Checkbutton(self.window, text="Transform st", variable=bvar, onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=2)
        self.bool_checkbox.append(bvar)

        bvar = BooleanVar()
        b = Checkbutton(self.window, text="Smooth", variable=bvar, onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=3)
        self.bool_checkbox.append(bvar)

        bvar = BooleanVar()
        b = Checkbutton(self.window, text="Random", variable=bvar, onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=4)
        self.bool_checkbox.append(bvar)
    
        for it, label, scale_range in zip(range(len(scale_labels)), scale_labels, from_to):
            value = DoubleVar() # dummy value
            s = Scale(self.window, resolution=0.1, variable=value, command=self.get_slider_value, label=label, length=75, width=15, from_=scale_range[0], to=scale_range[1])
            s.grid(row=off_y+1, column=it)
            self.scales_value.append(value)
        
        self.set_slider_value() # set to default settings
        self.get_slider_value() # get those values and set PID setting to those values
        self.set_checkbox_value()
        self.get_checkbox_value()

        self.window.increment(space=2) # add 1 to the client counter on the screen and add some offset to avoid overlapping

    def respawn(self):
        self.client_class.reset_car()

    def get_slider_value(self, v=0):
        values = []
        for scale in self.scales_value:
            values.append(scale.get())

        self.client_class.PID_settings = values

    def get_checkbox_value(self, v=0):
        bools = []
        for box in self.bool_checkbox:
            bools.append(box.get())

        self.client_class.loop_settings = bools

    def set_slider_value(self):
        assert len(self.client_class.PID_settings) == len(self.scales_value)

        for value, scale in zip(self.client_class.PID_settings, self.scales_value):
            scale.set(value)
    
    def set_checkbox_value(self):
        assert len(self.client_class.loop_settings) == len(self.bool_checkbox)

        for value, box in zip(self.client_class.loop_settings, self.bool_checkbox):
            box.set(value)

    def set_default(self, default=[]): # TODO
        pass

if __name__ == "__main__":
    from run_client import manual_client

    window = windowInterface()

    client = manual_client(window, "")
    auto = AutoInterface(window, client)
    client2 = manual_client(window, "")
    auto2 = AutoInterface(window, client2)

    window.mainloop() # display only at the end your window
