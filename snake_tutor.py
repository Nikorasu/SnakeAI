#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
from time import sleep

# This is a simplified version of the classic Snake game, reworked to play itself using depth-first search!
# This version will be designed to collect data on high scoring games, in a format easy to feed into a neural network.
# Highest score I've seen thus far: 50/64
# Built using MiniSnakes - https://github.com/eliasffyksen/MiniSnakes

cycles = 1000  # number of games to play
game_size = 8  # has to be same size as version NN plays
slowmode = False  # slows down the game so you can see what's going on

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
    #if max(scores) != -10: # if not trapped
    result = explore_path(bestsnake, depth + 1, max_depth)
    if result == None and depth != 0: # meaning got trapped in future
        return None
    if depth == 0 and result == None: # if path leads to trap, try next best
        nextaction = scores.index(max(scores, key=lambda x: x != max(scores)))
        bestaction = nextaction if scores[nextaction] != -10 else bestaction
    
    return bestaction

def single_bot_game(size=8, highscore=0):
    snake = t.zeros((size, size), dtype=t.int)
    snake[0, :4] = T([1,2,3, -1])
    reward = do(snake, 1)  # snake needs to grab first food so random food spawns
    print_state(snake)
    print(f"{reward:<6}{snake.max().item()-4:^6}{highscore:>6}{cycles:>9}")

    while reward != -10:
        if slowmode: sleep(0.1)
        #futures = [snake.clone() for _ in range(3)]
        #scores = [do(future, i) for i, future in enumerate(futures)]
        #snake = futures[scores.index(max(scores))]
        best_action = explore_path(snake)
        reward = do(snake, best_action) if best_action != None else -10
        print_state(snake)
        print(f"{reward:<6}{snake.max().item()-4:^6}{highscore:>6}{cycles:>9}")
        
    return snake.max().item()-4

if __name__ == '__main__':
    highscore = 0
    while highscore < game_size**2 - 3 or cycles > 0:
        highscore = max(highscore, single_bot_game(game_size, highscore))
        cycles -= 1