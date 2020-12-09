import numpy as np
import torchvision.transforms.functional as TF
import torch

from .classifier import load_model as load_classifier
from .detector import load_model as load_detector

class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""
    
    def __init__(self, player_id = 0):
        def get_goal(player_id):
            team = int(player_id % 2 == 0)
            if team == 0:
                enemy_goal = ([-10.45, 0.07, -64.5], [10.45, 0.07, -64.5])
                enemy_mid_goal = [0, 0.07, -64.5]
                our_goal = ([10.45, 0.07, 64.5], [-10.51, 0.07, 64.5])
                our_mid_goal = [0, 0.07, 64.5]
            else:
                our_goal = ([-10.45, 0.07, -64.5], [10.45, 0.07, -64.5])
                our_mid_goal = [0, 0.07, -64.5]
                enemy_goal = ([10.45, 0.07, 64.5], [-10.51, 0.07, 64.5])
                enemy_mid_goal = [0, 0.07, 64.5]

            return our_goal, our_mid_goal, enemy_goal, enemy_mid_goal
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.kart = "tux"
        self.player_id = player_id
        self.our_goal, self.our_mid_goal, self.enemy_goal, self.enemy_mid_goal = get_goal(self.player_id)

        # load model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.classifier = load_classifier().to(self.device)
        self.detector = load_detector().to(self.device)
        self.classifier.eval()
        self.detector.eval()
        
    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """
        ball_in_view = self.classifier(TF.to_tensor(image)[None].to(self.device)).cpu().tolist()
        x,y = (0,0)
        if ball_in_view[0][0] < ball_in_view[0][1]:
            loc, distance = self.detector(TF.to_tensor(image)[None].to(self.device))
            x,y = loc.cpu().squeeze();
            action['acceleration'] = 1
            action['steer'] = min(max(x*3, -1), 1)
        else:
            action['acceleration'] = 0
            action['brake'] = True
            action['steer'] = -1

        return action, ball_in_view[0][0] < ball_in_view[0][1], [x,y]

   
