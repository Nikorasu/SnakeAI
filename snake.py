#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel

# This is a simplified version of the classic Snake game, reworked specifically for implementing into machine learning.
# I've added a reward system to the do() function's return. Reward increases up to 9 the closer snake gets to food, moving away returns the negative.
# When food is eaten, that will reward 10. Game over indicated by a reward of -10, score is snake.max().item()-4
# The neural network inputs will correspond directly to the snake tensor, which I think is 8 by 8, so 64 input neurons?
# The neural network outputs will correspond to the available actions (0, 1, 2) for left, forward, right.
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
    return 10 if segs > prevsegs else (max(int(10-distaf),1) if distaf < distb4 else min(int(-(10-distaf)),-1))

def getdists(snake):
    head = divmod(t.argmax(snake).item(), snake.shape[1])
    food = divmod(t.argmin(snake).item(), snake.shape[1])
    return t.dist(t.tensor(head, dtype=t.float), t.tensor(food, dtype=t.float)).item()

def print_state(snake):
    for row in snake:
        row_str = ''.join([f"{value:2}" for value in row.tolist()])
        print(row_str)

# The neural network agent will have to initialize this stuff too, and handle the loop.
if __name__ == '__main__':
    board_size = 8
    snake = t.zeros((board_size, board_size), dtype=t.int)
    snake[0, :4] = T([1, 2, 3, -1]) # snake starts off 4 long (after next line), so NN learns not to crash into self early.
    score = do(snake, 1)  # snake needs to grab first food so random food spawns
    print()
    print_state(snake)
    print()

    while score != -10:
        action = input("Enter action (0: left, 1: forward, 2: right): ")
        print()
        score = do(snake, int(action) if action != '' else 1)
        print_state(snake)
        print(score)

    print('Score:', snake.max().item()-4)