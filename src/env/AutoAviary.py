import os
import sys
from datetime import datetime
import numpy as np
from gymnasium import spaces
import pybullet as p
import pkg_resources

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType

class AutoAviary(BaseAviary):
    """Multi-drone environment class for control applications + correct video"""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=True,
                 output_folder='auto_results'
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         vision_attributes=vision_attributes,
                         output_folder=output_folder
                         )


        self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if self.RECORD:
            # self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            # self.IMG_RES = np.array([1280, 720])
            self.IMG_RES = np.array([640, 480])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.PYB_STEPS_PER_CTRL != 0:
                print("\n\n[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            for i in range(self.NUM_DRONES):
                os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/"), exist_ok=True)
    ################################################################################

    
    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        # if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
        #     [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
        #                                              height=self.VID_HEIGHT,
        #                                              shadow=1,
        #                                              viewMatrix=self.CAM_VIEW,
        #                                              projectionMatrix=self.CAM_PRO,
        #                                              renderer=p.ER_TINY_RENDERER,
        #                                              flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        #                                              physicsClientId=self.CLIENT
        #                                              )
        #     (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # self.FRAME_NUM += 1
            # if self.VISION_ATTR:
            #     # print("A\n"*20)
            #     for i in range(self.NUM_DRONES):
            #         # print("B\n"*20)
            #         self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
            #         #### Printing observation to PNG frames example ############
            #         self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
            #                         img_input=self.rgb[i],
            #                         path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
            #                         frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
            #                         )
        if self.VISION_ATTR: 
                for i in range(self.NUM_DRONES):
                    # print("B\n"*20)
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )

        # #### Save PNG video frames if RECORD=True and GUI=False ####
        # if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
        #     [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
        #                                              height=self.VID_HEIGHT,
        #                                              shadow=1,
        #                                              viewMatrix=self.CAM_VIEW,
        #                                              projectionMatrix=self.CAM_PRO,
        #                                              renderer=p.ER_TINY_RENDERER,
        #                                              flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        #                                              physicsClientId=self.CLIENT
        #                                              )
        #     (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
        #     #### Save the depth or segmentation view instead #######
        #     # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
        #     # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
        #     # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
        #     # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
        #     self.FRAME_NUM += 1
        """ """



        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            self._saveLastAction(action)
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info
    
    
    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool=True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray 
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        # print(f"orig : {rot_mat}")
        # rot_mat = np.array([[1, 0,0],
        #                     [0.1,-1,0.2],
        #                     [0.01,0.2,-1]]).reshape(3, 3)
        # print(f"China : {rot_mat}")
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])

        """
        cameraEyePosition=self.pos[nth_drone, :]+np.array([0, 0, self.L]) - декартово положени камеры - позиция дрона + сдвиг 
        если сделать сдвиг (-1, 0, self.L), то будет вид сзади на дрона - выглядит здорово

        cameraTargetPosition = target = rot_mat + drone pos == позиция камеры

        cameraUpVector=[0, 0, 1] - угол камеры, если будет cameraUpVector=[0, 0, -1], то изображение вверх ногами
        """
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :]+np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        '''
        fov - угло обзора
        aspect - тоже угол обзора
        nearVal - ?
        farVal - дальность прорисовки
        '''
        # DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=10.0,
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0, 
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT,
                                                #  lightColor=[0, 0, 250],
                                                #  lightDirection=[2.2, 2.2, .5],
                                                #  lightDistance = 10, 
                                                #  lightAmbientCoeff = 1,
                                                #  lightDiffuseCoeff = 2, 
                                                 lightSpecularCoeff = -10, 
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg
    
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    def _addObstacles(self):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        path_3d_models = os.path.dirname(SCRIPT_DIR) + '/env/3d_models/'
        path_testures = os.path.dirname(SCRIPT_DIR) + '/env/testures/'
        if 1:
            # p.loadURDF('/home/oumuamua/sirius/env/3d_models/cube_with_sobaken.urdf',
            p.loadURDF(f'{path_3d_models}cube_with_sobaken.urdf',
                   [1.5, 1.5, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
            
            c = p.loadURDF('cube.urdf',
					[-1.1, 1, .5],					
					p.getQuaternionFromEuler([0, 0, 0]),
					globalScaling=1,					
					physicsClientId=self.CLIENT
					)
            x = p.loadTexture(f'{path_testures}tag.png')

            p.changeVisualShape(c, -1, textureUniqueId=x)



            cube = p.loadURDF('cube.urdf',
					[0, -1, 1.5],					
					p.getQuaternionFromEuler([0, 0, 0]),
					globalScaling=1,					
					physicsClientId=self.CLIENT
					)
            texture_1385 = p.loadTexture(f'{path_testures}april_36h11-252.png')
            print(texture_1385)
            p.changeVisualShape(cube, -1, textureUniqueId=texture_1385)
            # exit(10)
# -1 value is "link index"

#(link index has something to do with "joints", of which the cube apparently has none)
            # p.loadURDF("block.urdf",
            #            [.1, 0, .1],
            #            p.getQuaternionFromEuler([0, 0, 0]),
            #            physicsClientId=self.CLIENT
            #            )
            # p.loadURDF("cube_small.urdf",
            #            [0, .1, .1],
            #            p.getQuaternionFromEuler([0, 0, 0]),
            #            physicsClientId=self.CLIENT
            #            )
            # p.loadURDF("duck_vhacd.urdf",
            #            [-.1, 0, .1],
            #            p.getQuaternionFromEuler([0, 0, 0]),
            #            physicsClientId=self.CLIENT
            #            )
            # p.loadURDF("teddy_vhacd.urdf",
            #            [0, -.1, .1],
            #            p.getQuaternionFromEuler([10, 10, 0]),
            #            physicsClientId=self.CLIENT
            #            )
            # p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/urdfs_for_obstacles/monkey.urdf'),
            #        [0, -.1, .55],
            #        p.getQuaternionFromEuler([0, 0, 0]),
            #        physicsClientId=self.CLIENT
            #        )
        else:
            pass
