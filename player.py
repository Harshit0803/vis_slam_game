from vis_nav_game import Player, Action
import pygame
import cv2
# import os, shutil
from VLAD_interface import VLAD
import tracker
import numpy as np


vlad = VLAD()

sample_rate = 2 
action_hist =[]

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super(KeyboardPlayerPyGame, self).__init__()
        self.count = 0
        self.Phase = 1
        self.index = -1 
        self.target_loc = []
        self.train_imgs = []
        self.pose_hist = []
        self.pose = None
        self.AUTO_NAV = True  #change this to enable/disable NAV

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }


    
    def pre_navigation(self) -> None:
        
        if self.last_act is Action.QUIT:
            print("******************************************PRE-NAVIGATION********************************************************************")
            vlad.train(self.train_imgs)
            self.target_loc = vlad.query()
            target_locations = []
            # for target in self.target_loc:
            print("-------------> targets are ",self.target_loc,len(self.pose_hist))
            
            for target in self.target_loc:
                target_locations.append(self.pose_hist[sample_rate*target])
            tracker.reset(target_locations)
        return super().pre_navigation()

    def act(self):
        self.count+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.Phase+=1
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        if self.last_act is not Action.IDLE :
            action_hist.append(self.last_act)
            cur_x,cur_z = tracker.getOdometryFromOpticalFlow(cv2.cvtColor(self.fpv, cv2.COLOR_RGB2GRAY))
            self.pose = [cur_x,cur_z]
            self.pose_hist.append(self.pose)
            if len(action_hist)%sample_rate is 0:
                # filename = str(len(action_hist)//sample_rate)
                # cv2.imwrite('src/data/'+filename+'_img.png', self.fpv)
                self.train_imgs.append(self.fpv)

        state = self.get_state()
        if state is not None:
            Phase = state[1]
            if Phase is Phase.NAVIGATION and self.index<=sample_rate*self.target_loc[0] and self.AUTO_NAV:
                self.index+=1
                cur_x,cur_z = tracker.getOdometryFromOpticalFlow(cv2.cvtColor(self.fpv, cv2.COLOR_RGB2GRAY))
                return action_hist[self.index]
        if(self.last_act is not Action.IDLE):    
            cur_x,cur_z = tracker.getOdometryFromOpticalFlow(cv2.cvtColor(self.fpv, cv2.COLOR_RGB2GRAY))  
            self.pose = [cur_x,cur_z]  
        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        for index, img in enumerate(targets):
            filename = str(index)
            cv2.imwrite('/home/harshit/vis_nav_player/finGame/src/queries/'+filename+'_img.png', img)

        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def see(self, fpv):
        
        if fpv is None or len(fpv.shape) < 3:
            return
        


        self.fpv = fpv
            
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())