#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
from time import sleep

# This is a simplified version of the classic Snake game, reworked to play
# itself without a neural network, just a dumb bot picking best options. Scores around 35-40!
# Based on MiniSnakes - https://github.com/eliasffyksen/MiniSnakes

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
    return 10 if segs > prevsegs else (int(10-distaf) if distaf < distb4 else min(int(-(10-distaf)),-1))

def getdists(snake):
    head = divmod(t.argmax(snake).item(), snake.shape[1])
    food = divmod(t.argmin(snake).item(), snake.shape[1])
    return t.dist(t.tensor(head, dtype=t.float), t.tensor(food, dtype=t.float)).item()

def print_state(snake):
    for row in snake:
        row_str = ''.join([f"{value:2}" for value in row.tolist()])
        print(row_str)

def single_bot_game(highscore=0):
    board_size = 8
    snake = t.zeros((board_size, board_size), dtype=t.int)
    snake[0, :3] = T([1, 2,-1])
    reward = do(snake, 1)  # snake needs to grab first food so random food spawns

    print_state(snake)
    print(f"{reward:<5}{snake.max().item():^6}{highscore:>5}")

    while reward != -10:
        sleep(0.1)
        futures = [snake.clone() for _ in range(3)]
        scores = [do(future, i) for i, future in enumerate(futures)]
        snake = futures[scores.index(max(scores))]
        reward = max(scores)
        
        print_state(snake)
        print(f"{reward:<5}{snake.max().item():^6}{highscore:>5}")
        
    return snake.max().item()-3

if __name__ == '__main__':
    highscore = 0
    while highscore < 61:
        highscore = max(highscore, single_bot_game(highscore))
