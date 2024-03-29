import json
import time
import logging
import pprint

from gym_donkeycar.core.sim_client import SDClient



class PrivateAPIClient(SDClient):

    def __init__(self, address, private_key, poll_socket_sleep_time=0.01):
        super().__init__(*address, poll_socket_sleep_time=poll_socket_sleep_time)

        self.private_key = private_key
        self.is_verified = False

        self.raceSummary = {}

    def on_msg_recv(self, json_packet):
        msg_type = json_packet.get('msg_type')
        if msg_type == 'verified':
            self.is_verified = True

        elif msg_type == 'collision_with_starting_line':
            self.on_crossing_line(json_packet)

        elif msg_type == 'collision_with_cone':
            self.on_colliding_cone(json_packet)

        elif json_packet != {}:
            pprint.pprint(json_packet)

    def send_verify(self):
        msg = {"msg_type": "verify",
               "private_key": str(self.private_key)}
        self.send_now(json.dumps(msg))

    def send_seed(self, seed):
        msg = {"msg_type": "set_random_seed",
               "seed": str(seed)}
        self.send_now(json.dumps(msg))

    def on_crossing_line(self, json_packet):
        car_name = json_packet['car_name']

        if car_name in self.raceSummary:
            timeStamp = json_packet['timeStamp']
            lastTime = self.raceSummary[car_name]['lastCrossingLine']
            self.raceSummary[car_name]['lapTimes'].append(timeStamp-lastTime)
            self.raceSummary[car_name]['lastCrossingLine'] = timeStamp

        else:
            self.raceSummary[car_name] = {}
            self.raceSummary[car_name]['lapTimes'] = []
            self.raceSummary[car_name]['lastCrossingLine'] = json_packet['timeStamp']

    def on_colliding_cone(self, json_packet):
        car_name = json_packet['car_name']

        if car_name in self.raceSummary and len(self.raceSummary[car_name]['lapTimes']) > 0:
            self.raceSummary[car_name]['lapTimes'][-1] += 1

    def on_reset(self):
        self.raceSummary = {}


def test_clients():
    logging.basicConfig(level=logging.DEBUG)

    # test params
    host = "127.0.0.1"  # "donkey-sim.roboticist.dev"
    port = 9092
    # please enter your private key (provided in the menu of the simulator)
    private_key = "93618961"

    client = PrivateAPIClient((host, port), private_key)
    client.send_verify()

    time.sleep(0.5)  # wait for the response of the server
    print(f"is the client verified ? {client.is_verified}")

    # client.send_seed(42)  # try to set the seed used for the challenges

    while True:
        msg = input("")
        print(client.raceSummary)

        if msg == "reset":
            print("#########")
            client.on_reset()


if __name__ == "__main__":
    test_clients()
