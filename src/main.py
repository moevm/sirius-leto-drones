"""
Example
-------
In a terminal, run as:

    $ python <name>.py #запуск как обычной программы
    $ python <name>.py --vision_attributes True --gui False  # нет gui, зато есть видео с дрона 
    $ python <name>.py --vision_attributes True  # запуск видео + стандартная gui (сильно лагает, видео делается медленне)

    python3 sym_tag_v2.py --vision_attributes True --gui False
"""
import time
import argparse

import cv2
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from env.AutoAviary import AutoAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

# import gym_pybullet_drones.utils.position_commads as pc
from utils.tag_detector import detect_apriltags
from utils.area_check import area_check


DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48

# DEFAULT_SIMULATION_FREQ_HZ = 480
# DEFAULT_CONTROL_FREQ_HZ = 96

DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEF_VISION_ATTR = False


def go_back(tmp_pos, tmp_rpy, delta):
    target_pos = tmp_pos.copy()
    target_pos[0] += delta * np.cos(tmp_rpy[2])
    target_pos[1] -= delta * np.sin(tmp_rpy[2])
    return target_pos

def go_forward(tmp_pos, tmp_rpy, delta):
    target_pos = tmp_pos.copy()
    target_pos[0] -= delta * np.cos(tmp_rpy[2])
    target_pos[1] += delta * np.sin(tmp_rpy[2])
    return target_pos

def go_left(tmp_pos, tmp_rpy, delta):
    target_pos = tmp_pos.copy()
    target_pos[1] += delta * np.cos(tmp_rpy[2])
    target_pos[0] -= delta * np.sin(tmp_rpy[2])
    return target_pos

def go_right(tmp_pos, tmp_rpy, delta):
    target_pos = tmp_pos.copy()
    target_pos[1] -= delta * np.cos(tmp_rpy[2])
    target_pos[0] += delta * np.sin(tmp_rpy[2])
    return target_pos

def clockwise(tmp_rpy, delta):
    target_rpy = tmp_rpy.copy()
    target_rpy[2] -= delta
    return target_rpy

def counterclockwise(tmp_rpy, delta):
    target_rpy = tmp_rpy.copy()
    target_rpy[2] += delta
    return target_rpy

def go_down(tmp_pos, delta):
    target_pos = tmp_pos.copy()
    if target_pos[2] > 0.01:
        target_pos[2] -= delta
    return target_pos

    


 
def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        vision_attributes = DEF_VISION_ATTR,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):

    #### Initialize the trajectories ###########################
    PERIOD = duration_sec
    NUM_WP = control_freq_hz*PERIOD
    
    # x = -1
    # y = -1
    # z = 0.5

    x = -1
    y = -7
    z = 0.5

    INIT_XYZS = np.array([[x, y, z]])
    env = AutoAviary(drone_model=drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB_GND,
                     # physics=Physics.PYB, # For comparison
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=True,
                     vision_attributes = vision_attributes
                     )


    wp_counters = [0]
    wp_counter = 0

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    duration_sec=duration_sec,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=drone)]

    tmp_pos = [x, y, z]
    tmp_rpy = [0, 0, 0]
    target_pos = [None, None, None]
    target_rpy = [None, None, None]

    last_area = 0

    i = 0
    #### Run the simulation ####################################
    action = np.zeros((1,4))
    START = time.time()
    while True:
        # print(f'\n\n{i}\n\n')

        obs, reward, terminated, truncated, info = env.step(action)
        
        if not i: 
            target_pos = tmp_pos
            target_rpy = [0, 0, 0]

        # текущий шаг присваеваем как желаемый на поршлом ходу
        tmp_pos = target_pos
        tmp_rpy = target_rpy

        delta = 0.03
        delta_rpy = 0.01

        # продумываем последующий шаг

        if env.VISION_ATTR:
            drone_img = env.rgb[0].astype(np.uint8)
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_RGBA2BGR)

            drone_img, centers, areas, id_tags = detect_apriltags(drone_img, True)

            if len(centers) != 0:
                for area in areas:  
                    ratio = area_check(area)
                    last_area = ratio
                    print(f'\n{i} - {ratio}\n')
                
                for center in centers:
                    if (np.abs(center[1] - drone_img.shape[0]//2) ) < 20:
                        print(f"Center in center: {(np.abs(center[0] - drone_img.shape[0]//2) - 70)}")  
                        print(drone_img.shape[0]//2)  
                        # delta = 0.05
                        target_pos = go_forward(tmp_pos, tmp_rpy, delta)
                    elif ((center[1] - drone_img.shape[0]//2) ) < 0:
                        # delta = 0.05
                        # go_left(tmp_pos, tmp_rpy, delta)
                        target_rpy = counterclockwise(tmp_rpy, delta_rpy/1.3)
                        target_pos = go_forward(tmp_pos, tmp_rpy, delta / 2)
                        print("\nLEFT\n"*5)
                        print(drone_img.shape[0]//2)  
                    else:
                        # delta = 0.05
                        # go_right(tmp_pos, tmp_rpy, delta)
                        target_rpy = clockwise(tmp_rpy, delta_rpy/1.3)
                        target_pos = go_forward(tmp_pos, tmp_rpy, delta / 2)
                        print("\nRIGHT\n"*5)
                        print(drone_img.shape[0]//2)  
            else:

                if last_area > 50: 
                    print("Садимся")
                    target_pos = go_down(tmp_pos, delta )
                elif last_area > 50:
                    print("Садимся")
                elif last_area < 5:
                    delta = 0.01
                    target_rpy = counterclockwise(tmp_rpy, delta)
                    print("No tags in image")
                else:
                    rng = np.random.default_rng()
                    tmp_delta = rng.integers(low=0, high=10) / 100 
                    print(f'Случайное движение при потере тыга ({tmp_delta})')  
                    target_pos += tmp_delta


            cv2.imwrite(f'./screenshots/frame_{i}.png', drone_img)
           

        action[0], _, _ = ctrl[0].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                             state=obs[0],
                                                             target_pos=target_pos,
                                                             target_rpy=target_rpy
                                                             )
        i += 1
        if wp_counter < NUM_WP - 1:
            wp_counter = wp_counter + 1
        else:
            wp_counter = 0



        # # #### Log the simulation ####################################
        for j in range(1):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j]
                    #    control=np.hstack([TARGET_POS[wp_counter, :], INIT_XYZS[j ,2], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--vision_attributes',  default=DEF_VISION_ATTR,      type=str2bool,      help='Whether to record a video frome drone (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
