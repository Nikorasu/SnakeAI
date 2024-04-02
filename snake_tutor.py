#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
from time import sleep

# This is a simplified version of the classic Snake game, reworked to play itself using depth-first search!
# This version will be designed to collect data on high scoring games, in a format easy to feed into a neural network.
# Built using MiniSnakes - https://github.com/eliasffyksen/MiniSnakes
# hs 52/64

slowmode = False   # slows things down so you can watch what's going on
num2save = 500     # number of high-scoring games to save (actual turn count may vary)
maxgames = 20000   # maximum number of games to play before giving up

game_size = 8      # has to be same size as version NN plays
threshold = 40     # threshold over which to save games, locked in this version

def do(snake: t.Tensor, action: int):
    prevsegs = snake.max().item()
    distb4 = getdists(snake)
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in positions]
    rotation = T([[0, -1], [1, 0]]).matrix_power(3 + action)
    pos_next = (pos_cur + (pos_cur - pos_prev) @ rotation) % T(snake.shape)
    
    if (snake[tuple(pos_next)] > 0).any():
        return -10
    
    if snake[tuple(pos_next)] == -1:
        pos_food = (snake == 0).flatten().to(t.float).multinomial(1)[0]
        snake[unravel(pos_food, snake.shape)] = -1
    else:
        snake[snake > 0] -= 1

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1
    
    segs = snake.max().item()
    distaf = getdists(snake)
    return 10 if segs > prevsegs else (max(int(10-distaf),1) if distaf < distb4 else min(int(-(10-distaf)),-1))

def getdists(snake):
    head = divmod(t.argmax(snake).item(), snake.shape[1])
    food = divmod(t.argmin(snake).item(), snake.shape[1])
    return t.dist(t.tensor(head, dtype=t.float), t.tensor(food, dtype=t.float)).item()

def print_state(snake):
    for row in snake:
        row_str = ''.join([f"{value:2}" for value in row.tolist()])
        print(row_str)

def explore_path(snake, depth=0, max_depth=50):
    futures = [snake.clone() for _ in range(3)]
    scores = [do(future, i) for i, future in enumerate(futures)]
    result = bestaction = scores.index(max(scores))
    bestsnake = futures[scores.index(max(scores))]
    
    if max(scores) == 10 or depth >= max_depth: # if food or max depth reached
        return bestaction
    if max(scores) == -10 and depth != 0: # if trapped now, and not at depth 0
        return None
    result = explore_path(bestsnake, depth + 1, max_depth)
    if result == None and depth != 0: # meaning got trapped in future
        return None
    if depth == 0 and result == None: # if path leads to trap, try next best
        nextaction = scores.index(max(scores, key=lambda x: x != max(scores)))
        bestaction = nextaction if scores[nextaction] != -10 else bestaction
    
    return bestaction

def trimdeath(game_data):
    tens = 0  # this function tries to remove actions that lead up to death
    for i in range(len(game_data)-1, -1, -1):
        if game_data[i][2] == 10:
            tens += 1
            if tens == 2:
                return game_data[:i+1]
    return game_data

class GameRecorder:
    def __init__(self):
        self.cycles = maxgames
        self.bestgames_cache = []
        self.games_collected = 0
        self.turnspergame = []
        self.scores = []
        self.highscore = 0

    def single_bot_game(self):
        game_data = []
        turns = 0
        snake = t.zeros((game_size, game_size), dtype=t.int)
        snake[0, :4] = T([1,2,3, -1])
        reward = do(snake, 1) # snake needs to grab first food so random food spawns

        while reward != -10:
            turns += 1
            state = snake.clone()
            best_action = explore_path(snake)
            reward = do(snake, best_action) if best_action != None else -10
            if slowmode:
                sleep(0.2)
                print_state(snake)
                print(f"{reward:<6}{snake.max().item()-4:^6}{self.highscore:>6}{1+maxgames-self.cycles:>9}")
            if turns > game_size: # avoid saving some of the early easy moves
                game_data.append([state, best_action, reward, snake.clone()]) # state, action, reward, next_state
        
        print_state(snake)
        print(f"{snake.max().item()-4:<6}{self.highscore:^6}{1+maxgames-self.cycles:>9}")
        
        if snake.max().item()-4 >= threshold:
            game_data = trimdeath(game_data)
            self.games_collected += 1
            self.bestgames_cache.extend(game_data)
            self.scores.append(snake.max().item()-4)
            self.turnspergame.append(turns)
            print(f"Scored over {threshold}!  Saved: {self.games_collected}")
        
        return snake.max().item()-4
    
    def run(self):
        while self.games_collected < num2save and self.cycles > 0:
            self.highscore = max(self.highscore, self.single_bot_game())
            self.cycles -= 1

        print("\nDone!")
        print(f"Games collected:    {self.games_collected:>5}")
        print(f"Average Turns/game: {sum(self.turnspergame) / len(self.turnspergame):>5}")
        print(f"Total turns saved:  {len(self.bestgames_cache):>5}")
        print(f"Average score:      {sum(self.scores) / len(self.scores):>5}")
        print(f"Highest score:      {self.highscore:>5}")
        print("\nSaving to file..")

        t.save(self.bestgames_cache, f"snakedata_{self.games_collected}_{sum(self.scores) // len(self.scores)}.pt")

if __name__ == '__main__':
    tutor = GameRecorder()
    tutor.run()