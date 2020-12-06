import pystk
import numpy as np


class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        if save is not None:
            import PIL.Image
            import os
            if not os.path.exists(save):
                os.makedirs(save)
            save_ball = os.path.join(save,'with_Ball')
            if not os.path.exists(save_ball):
                os.makedirs(save_ball)
            save_no_ball = os.path.join(save,'without_Ball')
            if not os.path.exists(save_no_ball):
                os.makedirs(save_no_ball)

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()
            list_actions = []
            ball = state.soccer.ball
            for i, p in enumerate(self.active_players):
                player = state.players[i]
                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)
                # project ball onto the screen
                ball_loc_screen = world_to_screen(player, ball.location)
                # true if the ball is in view of this player
                ball_in_view = -1 < ball_loc_screen[0] < 1 and -1 < ball_loc_screen[1] < 1
                ball_distance = np.linalg.norm(np.array(player.kart.location) - np.array((ball.location)))

                if save is not None:
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    im = PIL.Image.fromarray(image)
                    
                    # draw the ball on image, remove during data collection
                    if ball_in_view:
                        # from PIL import ImageDraw
                        # H, W = image.shape[0], image.shape[1]
                        # draw = ImageDraw.Draw(im)
                        # ball_loc_image = ball_loc_screen
                        # ball_loc_image[0] = ball_loc_image[0] * (W/2) + W/2
                        # ball_loc_image[1] = ball_loc_image[1] * (H/2) + H/2
                        # draw.ellipse((ball_loc_image[0]-10, ball_loc_image[-1]-10, 
                  	     #     ball_loc_image[0]+10, ball_loc_image[-1]+10), outline='blue')
                        fn = os.path.join(save_ball, 'player%02d_%05d.png' % (i, t))
                        im.save(fn)

                        with open(fn + '.csv', 'w') as f:
                            f.write('%0.1f,%0.1f,%0.1f' % tuple(list(np.append(ball_loc_screen, ball_distance))))
                    else:
                        im.save(os.path.join(save_no_ball, 'player%02d_%05d.png' % (i, t)))
                    
                    
            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score#, state.players[0], state.soccer
        return state.soccer_score#, state.players[0], state.soccer

    def close(self):
        self.k.stop()
        del self.k

# convert world coordinate to screen coordinate with range = ([-1,1],[-1,1])
def world_to_screen(player, dest):
    proj = np.array(player.camera.projection).T
    view = np.array(player.camera.view).T
    p = proj @ view @ np.array(list(dest) + [1])
    screen =  np.array([p[0] / p[-1], - p[1] / p[-1]])
    return screen
