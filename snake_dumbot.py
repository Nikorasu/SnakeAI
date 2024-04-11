#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
from time import sleep

# This is a simplified version of the classic Snake game, reworked to play
# itself without a neural network, just a dumb bot picking best options. Scores around 35-40!
# Based on MiniSnakes - https://github.com/eliasffyksen/MiniSnakes

game_size = 10

def do(snake: t.Tensor, action: int):
    prevsegs = snake.max().item()  # to keep track of when food gets eaten
    distb4 = getdists(snake)  # to track whether snake gets closer to or away from food
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in positions]
    rotation = T([[0, -1], [1, 0]]).matrix_power(3 + action)
    pos_next = (pos_cur + (pos_cur - pos_prev) @ rotation) % T(snake.shape)
    
    if (snake[tuple(pos_next)] > 0).any():  # if the action will cause snake to die
        return -10
    
    if snake[tuple(pos_next)] == -1:  # if snake eats food
        pos_food = (snake == 0).flatten().to(t.float).multinomial(1)[0]
        snake[unravel(pos_food, snake.shape)] = -1
    else:
        snake[snake > 0] -= 1

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1
    
    segs = snake.max().item()
    distaf = getdists(snake)
    # if snake ate food (got longer), return 10, else return 1-9 depending on how close snake is to food, positive if closer, negative if further
    return 10 if segs > prevsegs else (max(int(10-distaf),1) if distaf < distb4 else min(int(-(10-distaf)),-1))

def getdists(snake):  # this function calculates the distance between the snake head and the food
    head = divmod(t.argmax(snake).item(), snake.shape[1])
    food = divmod(t.argmin(snake).item(), snake.shape[1])
    return t.dist(t.tensor(head, dtype=t.float), t.tensor(food, dtype=t.float)).item()

def print_state(snake):
    c = {'red': '\x1b[31m', 'green': '\x1b[32m', 'white': '\x1b[37m'}
    for row in snake:
        row_str = ''.join([f"{c['green'] if value>0 else c['red'] if value==-1 else c['white']}{value:2}" for value in row.tolist()])
        print(row_str, end='\x1b[0m\n')

def single_bot_game(size=10, highscore=0):
    snake = t.zeros((size, size), dtype=t.int)
    snake[0, :3] = T([1, 2,-1])  # sets up initial snake
    reward = do(snake, 1)  # before first turn, snake needs to eat initial food so first random food spawns
    print_state(snake)
    print(f"{reward:<7}{snake.max().item()-3:^7}{highscore:>7}")

    while reward != -10:
        sleep(0.1)
        futures = [snake.clone() for _ in range(3)] # clone the snake so we can check the 3 possible future states
        scores = [do(future, i) for i, future in enumerate(futures)] # get the score for each of the 3 choices (0 left, 1 forward, 2 right)
        snake = futures[scores.index(max(scores))] # set the snake to the best choice
        reward = max(scores)
        print_state(snake)
        print(f"{reward:<7}{snake.max().item()-3:^7}{highscore:>7}")
        
    return snake.max().item()-3


if __name__ == '__main__':
    highscore = 0
    while highscore < game_size**2 - 3:
        highscore = max(highscore, single_bot_game(game_size, highscore))
