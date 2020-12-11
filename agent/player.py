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
        self.last_ball_screen = [0,0]
        self.last_ball_world = [0,0,0]
        self.last_velocity = 0
        
        self.s = 0
        self.frame = 0
        # load model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.classifier = load_classifier().to(self.device)
        self.detector = load_detector().to(self.device)
        self.classifier.eval()
        self.detector.eval()
    
    

    def act(self, image, player_info):
        def world_to_screen(player, dest):
            proj = np.array(player.camera.projection).T
            view = np.array(player.camera.view).T
            p = proj @ view @ np.array(list(dest) + [1])
            screen =  np.array([p[0] / p[-1], - p[1] / p[-1]])
            return screen, p[-1] > 0

        def to_world(aim_point, player, height=0):
            proj = np.array(player.camera.projection).T
            view = np.array(player.camera.view).T
            pv_inv = np.linalg.pinv(proj @ view)
            xy, d = pv_inv.dot([aim_point[0],-aim_point[1],0,1]), pv_inv[:, 2]
            x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
            t = (height-x0[1]) / (x1[1] - x0[1])
            if t < 1e-3 or t > 10:
                # Project the point forward by a certain distance, if it would end up behind
                t = 10
            return t * x1 + (1-t) * x0

        def ball_in_front(ball_distance, ball_screen):
            # return ball_distance<7 and abs(ball_screen[0]) < 0.15
            return True

        def hit_ball(ball_screen, player, enemy_mid_goal):
            goal_screen, _ = world_to_screen(player, enemy_mid_goal)
            return ball_screen[0] - goal_screen[0]

        def clip_steer(steer):
            return min(max(steer, -1), 1)

        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        # print("========")
        # print("world coord: "+str(self.enemy_mid_goal))
        # test_screen, _ = world_to_screen(player_info, self.enemy_mid_goal)
        # print("screen coord: "+str(test_screen))
        # test_world = to_world(test_screen, player_info) 
        # print("estimated world coord: " +str(test_world))
        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        """
        Your code here.
        """
        
        ball_on_screen = self.classifier(TF.to_tensor(image)[None].to(self.device)).cpu().tolist()
        
        if(self.frame < 10):
            return action ball_on_screen[0][0] < ball_on_screen[0][1], self.last_ball_screen.copy()
        
        if ball_on_screen[0][0] < ball_on_screen[0][1]:
            
            o, _ = self.detector(TF.to_tensor(image)[None].to(self.device))
            ball_screen = o.cpu().squeeze().detach().numpy().tolist()
            
            self.last_ball_screen = ball_screen
            
            ball_world = to_world(ball_screen, player_info)
            
            self.s +=1
            velocity = (ball_world - self.last_ball_world)/self.s
            ball_acceleration = (velocity - self.last_velocity)/self.s
            self.s = 0
            angle = np.array(self.enemy_mid_goal - ball_world)
            angle = 1.5*angle/np.linalg.norm(angle)
            aim = ball_world + velocity + 1/2*ball_acceleration + angle
            
            self.last_ball_world = ball_world
            self.last_velocity = velocity
            
            # print(ball_world)
            ball_distance = min(np.linalg.norm(np.array(player_info.kart.location) - np.array(ball_world)),30)
            # print(ball_distance)
            
            action['acceleration'] = 1
            if ball_in_front(ball_distance, ball_screen):
                action['steer'] = clip_steer(hit_ball(aim, player_info, self.enemy_mid_goal))
            else:
                action['steer'] = clip_steer(ball_screen[0]*5)
        else:
            self.s+=1 
            action['acceleration'] = 0
            action['brake'] = True
            action['steer'] = -1 if self.last_ball_screen[0] > 0 else 1 
            # action['steer'] = -1
            # action['steer'] = -1 if np.random.uniform(-1,1) > 0 else 1

        return action, ball_on_screen[0][0] < ball_on_screen[0][1], self.last_ball_screen.copy()

   
