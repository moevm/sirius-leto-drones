"""
Example
-------
In a terminal, run as:

    $ python online.py #запуск как обычной программы
    $ python online.py --vision_attributes True --gui False # нет gui, зато есть видео с дрона 
    $ python online.py ----vision_attributes # запуск видео + стандартная gui (сильно лагает, видео делается медленне)


"""
import time
import argparse
import numpy as np
import cv2


from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from env.AutoAviary import AutoAviary
# from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

import gym_pybullet_drones.utils.position_commads as pc


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
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[.0, .0, .1]]) # WARNING!!!!!!!!
    

    #### Initialize the trajectories ###########################
    PERIOD = duration_sec
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3)) + 1.1     
    TARGET_RPY = np.zeros((NUM_WP, 3))
    INIT_XYZS = np.array([[TARGET_POS[0, 0], TARGET_POS[0, 1], TARGET_POS[0, 2]]])

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


    #### Run the simulation ####################################
    action = np.zeros((1,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Here we can process image
        # Тута мы можем обработать изображение
        # print(f"\n\n{env.rgb}\n\n")

        rgba = np.asarray(env.rgb[0])
        rgb = rgba[:, :, 0:3]
        bgr = rgb[... , :: -1]


        # cv2.imwrite(f"frame{i}.jpg", bgr)


        step_size = .01
        degree_step = 1/60
        if i < 100: #0-99 == 100 steps
            TARGET_POS[i+1, 1] = TARGET_POS[i, 1] 
            TARGET_POS[i+1, 2] = TARGET_POS[i, 2]
            if i % 100 > 20: # 21-99 == 80 steps
                TARGET_RPY[i+1, 2] = TARGET_RPY[i, 2] + degree_step
                TARGET_POS[i+1, 0] = TARGET_POS[i, 0] + step_size
        elif i < 200: 
            TARGET_POS[i+1, 0] = TARGET_POS[i, 0] 
            TARGET_POS[i+1, 1] = TARGET_POS[i, 1] 
            TARGET_POS[i+1, 2] = TARGET_POS[i, 2]
            if i % 100 > 20:
                TARGET_POS[i+1, 1] = TARGET_POS[i, 1] + step_size
        elif i < 300:
            TARGET_POS[i+1, 0] = TARGET_POS[i, 0] 
            TARGET_POS[i+1, 1] = TARGET_POS[i, 1] 
            TARGET_POS[i+1, 2] = TARGET_POS[i, 2]
            if i % 100 > 20:
                TARGET_POS[i+1, 0] = TARGET_POS[i, 0] - step_size
        elif i < 400:
            TARGET_POS[i+1, 0] = TARGET_POS[i, 0] 
            TARGET_POS[i+1, 1] = TARGET_POS[i, 1] 
            TARGET_POS[i+1, 2] = TARGET_POS[i, 2]
            if i % 100 > 20:
                TARGET_POS[i+1, 1] = TARGET_POS[i, 1] - step_size



        action[0], _, _ = ctrl[0].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                             state=obs[0],
                                                             target_pos=TARGET_POS[wp_counter, :],
                                                             target_rpy=TARGET_RPY[wp_counter, :]
                                                             )
    

        '''
        i == wp_counter 
        '''
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