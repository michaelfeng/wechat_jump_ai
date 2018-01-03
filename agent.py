import time
import wda
import math
import random
import itertools
import os
from PIL import Image, ImageDraw
import shutil

under_game_score_y = 200
piece_base_height_1_2 = 13
piece_body_width = 47
screenshot_backup_dir = 'screenshot_backups/'
class LearningAgent():

    def __init__(self, learning=False, epsilon=1.0, alpha=0.5, penalty=-0.05):
        self.valid_actions = [i/10 for i in range(4,12)]
        self.learning = learning
        self.Q = dict()
        self.epsilon = epsilon
        self.alpha = alpha
        self.penalty=penalty
        self.t = 1
        self.c = wda.Client('http://localhost:8100')
        self.s = self.c.session()
        self.x = 0
        self.y = 0
        # valid actions
        print('Valid actions:',self.valid_actions)
        print(self.Q)

    def reset(self, testing=False):
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.epsilon = math.exp(-0.005 * self.t)
            self.t += 1
        return

    def build_state(self):
        # the distance between start point and target point
        self.pull_screenshot()
        im = Image.open("./1.png")
        piece_x, piece_y, board_x, board_y = map(int, self.find_piece_and_board(im))
        self.x = board_x
        self.y = board_y
        print('location: ', piece_x, piece_y, board_x, board_y)
        state = math.sqrt(math.fabs(piece_x - board_x) ** 2 + math.fabs(piece_y - board_y) ** 2)
        state = int(state)
        return state

    def pull_screenshot(self):
        self.c.screenshot('1.png')

    def is_over(self):
        is_over = False
        # check if the game is over
        time.sleep(random.uniform(0.7, 1.1)) 
        #print("snapchat!!!")
        self.pull_screenshot()
        im = Image.open("./1.png")
        ts = int(time.time())
        piece_x, piece_y, board_x, board_y = map(int, self.find_piece_and_board(im))
        self.save_debug_creenshot(ts, im, piece_x, piece_y, board_x, board_y)
        self.backup_screenshot(ts)
        bias = int(math.sqrt(math.fabs(board_x - self.x) ** 2 + math.fabs(board_y - self.y) ** 2)/100)
        print("bias: ", bias)
        #print("last x: ",self.x)
        #print("last y: ", self.y)
        if (piece_x == 0 and piece_y == 0) or bias > 10:
            is_over = True
        self.x = piece_x
        self.y = piece_y
        '''
        print("new x:",piece_x)
        print("new y:",piece_y)
        print("new target x:", board_x)
        print("new target y:" ,board_y)'''
        return is_over, bias

    def get_maxQ(self, state):
        if state in self.Q:
            maxQ = max(self.Q[state].values())
        else:
            self.createQ(state)
            maxQ = max(self.Q[state].values())
        return maxQ

    def createQ(self, state):
        if self.learning:
            if not state in self.Q:
                self.Q.setdefault(state, {action: 0.0 for action in self.valid_actions})

    def choose_action(self, state):
        self.state = int(state)
        # default action 
        action = '0.5' 
        if not self.learning or random.random() <= self.epsilon:
            action = random.choice(self.valid_actions)
        else:
            maxQ = self.get_maxQ(state)
            maxQ_actions = [act for act,val in self.Q[state].items() if val == maxQ]
            print("maxQ_actions:", maxQ_actions)
            action = random.choice(maxQ_actions)
        return action

    def learn(self, state, action, reward):
        if self.learning:
            self.Q[state][action] = reward * self.alpha + self.Q[state][action] * (1 - self.alpha)

    def update(self, action, state, bias):
        #state = self.build_state()          # Get current state
        # Create 'state' in Q-table
        if state != 0:
            self.createQ(state)                 
        # Receive a reward
        reward = float("{0:.2f}".format(bias * self.penalty + 0.2))
        print("bias: ", bias)
        print("state: ", state)
        print('action: ', action)
        print('reward: ', reward)
        # Q-learn
        if state != 0:
            self.learn(state, action, reward)

    def restart(self):
        self.s.tap(185, 500)
        time.sleep(random.uniform(1, 1.1)) 

    def jump(self):
        print("## Start ...")
        state = self.build_state()
        action = self.choose_action(state)
        # tap hold
        # time.sleep(random.uniform(1, 1.1)) 
        self.s.tap_hold(85, 500, action)
        time.sleep(2)
        isOver, bias = self.is_over()
        print("isOver: ", isOver)
        if state == 0:
            bias = 1
        if not isOver:
            self.update(action, state, bias)
        else:
            bias = 5
            self.update(action, state, bias)
            time.sleep(random.uniform(1, 1.1)) 
        self.t += 1
        print("state:"+str(state)+",action:"+ str(action) + ",reward:"+str(float("{0:.2f}".format(bias * self.penalty + 0.2))))
        #print('self.Q')
        #for k,v in self.Q.items():
        #    print(k,v)
        return isOver
        
    def find_piece_and_board(self, im):
        w, h = im.size
        #print("size: {}, {}".format(w, h))

        piece_x_sum = 0
        piece_x_c = 0
        piece_y_max = 0
        board_x = 0
        board_y = 0
        scan_x_border = int(w / 8)
        scan_start_y = 0 
        im_pixel = im.load()


        for i in range(under_game_score_y, h, 50):
            last_pixel = im_pixel[0, i]
            for j in range(1, w):
                pixel = im_pixel[j, i]
                if pixel[0] != last_pixel[0] or pixel[1] != last_pixel[1] or pixel[2] != last_pixel[2]:
                    scan_start_y = i - 50
                    break
            if scan_start_y:
                break

        #print("scan_start_y: ", scan_start_y)

        for i in range(scan_start_y, int(h * 2 / 3)):
            for j in range(scan_x_border, w - scan_x_border):
                pixel = im_pixel[j, i]
                if (50 < pixel[0] < 60) and (53 < pixel[1] < 63) and (95 < pixel[2] < 110):
                    piece_x_sum += j
                    piece_x_c += 1
                    piece_y_max = max(i, piece_y_max)

        if not all((piece_x_sum, piece_x_c)):
            return 0, 0, 0, 0
        piece_x = piece_x_sum / piece_x_c
        piece_y = piece_y_max - piece_base_height_1_2

        for i in range (int (h / 3), int (h * 2 / 3)):
            last_pixel = im_pixel[0, i]
            if board_x or board_y:
                break
            board_x_sum = 0
            board_x_c = 0

            for j in range(w):
                pixel = im_pixel[j, i]
                if abs(j - piece_x) < piece_body_width:
                    continue

                if abs(pixel[0] - last_pixel[0]) + abs(pixel[1] - last_pixel[1]) + abs(pixel[2] - last_pixel[2]) > 10:
                    board_x_sum += j
                    board_x_c += 1

            if board_x_sum:
                board_x = board_x_sum / board_x_c

        board_y = piece_y - abs(board_x - piece_x) * math.sqrt(3) / 3

        if not all((board_x, board_y)):
            return 0, 0, 0, 0
        return piece_x, piece_y, board_x, board_y


    def backup_screenshot(self,ts):
        if not os.path.isdir(screenshot_backup_dir):
            os.mkdir(screenshot_backup_dir)
        shutil.copy('1.png', '{}{}.png'.format(screenshot_backup_dir, ts))


    def save_debug_creenshot(self,ts, im, piece_x, piece_y, board_x, board_y):
        draw = ImageDraw.Draw(im)
        draw.line((piece_x, piece_y) + (board_x, board_y), fill=2, width=3)
        draw.line((piece_x, 0, piece_x, im.size[1]), fill=(255, 0, 0))
        draw.line((0, piece_y, im.size[0], piece_y), fill=(255, 0, 0))
        draw.line((board_x, 0, board_x, im.size[1]), fill=(0, 0, 255))
        draw.line((0, board_y, im.size[0], board_y), fill=(0, 0, 255))
        draw.ellipse((piece_x - 10, piece_y - 10, piece_x + 10, piece_y + 10), fill=(255, 0, 0))
        draw.ellipse((board_x - 10, board_y - 10, board_x + 10, board_y + 10), fill=(0, 0, 255))
        del draw
        im.save('{}{}_d.png'.format(screenshot_backup_dir, ts))

def run():        
    # Create agent
    agent = LearningAgent(learning=True)
    n_test = 5
    n_train = 2000
    tolerance=0.005
    log_metrics = True
    if agent.learning:
        log_filename = os.path.join("logs", "sim_improved-learning.csv")        
    else:
        log_filename = os.path.join("logs", "sim_no-learning.csv")
    table_filename = os.path.join("logs","sim_improved-learning.txt")    
    table_file = open(table_filename, 'w')
    log_file = open(log_filename, 'w')
    total_trials = 1
    testing = False
    trial = 1
    # Run 
    while True:
        print("trial:",trial)
        print("total_trials:",total_trials)

        if testing:
            if trial > n_test:
                break
        else:
            if trial > n_train:
                testing = True
                trial = 1
        
        # Pretty print to terminal                                                                
        print()
        print("/-------------------------")
        if testing:
            print("| Testing trial {}".format(trial))
        else:
            print("| Training trial {}".format(trial))

        print("\-------------------------")
        print()
        # Increment                                                                                 
        trial += 1
        total_trials = total_trials + 1

        current_time = 0.0
        last_updated = 0.0
        start_time = time.time()
        #current_time = time.time() - start_time
        isOver = False
        while not isOver:
            isOver = agent.jump()
            agent.reset(testing)
            if isOver:
                agent.restart()

        # Clean up
        if log_metrics:
            if agent.learning:
                f = table_file

                f.write("/-----------------------------------------\n")
                f.write("| State-action rewards from Q-Learning\n")
                f.write("\-----------------------------------------\n\n")

                for state in agent.Q:
                    f.write("{}\n".format(state))
                    for action, reward in agent.Q[state].items():
                        f.write(" -- {} : {:.2f}\n".format(action, reward))
                    f.write("\n")  
        print("\nSimulation ended. . . ")
    table_file.close()
    log_file.close()

if __name__ == "__main__":
    run()
    

