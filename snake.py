#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel

# This is a simplified version of the classic Snake game, reworked specifically for implementing into machine learning.
# I've added a reward system to the do() function's return. Still adjusting how things are rewarded tho.
# When food is eaten, that will reward 10. Game over indicated by a reward of -1, final score is snake.max().item()-3
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
        return -1 #snake.max().item() #snake[tuple(pos_cur)].item()
    
    if snake[tuple(pos_next)] == -1:
        pos_food = (snake == 0).flatten().to(t.float).multinomial(1)[0]
        snake[unravel(pos_food, snake.shape)] = -1
    else:
        snake[snake > 0] -= 1

    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1
    
    segs = snake.max().item()
    distaf = getdists(snake)
    return 10 if segs > prevsegs else int(distaf < distb4)#0  #(1 if distaf < distb4 else -1) #int(distaf < distb4)

def getdists(snake):
    head = divmod(t.argmax(snake).item(), snake.shape[1]) #(snake == t.max(snake)).nonzero(as_tuple=True)
    food = divmod(t.argmin(snake).item(), snake.shape[1])
    return t.dist(t.tensor(head, dtype=t.float), t.tensor(food, dtype=t.float)).item() #t.cdist(). .unsqueeze(0)

def print_state(snake):
    for row in snake:
        row_str = ''.join([f"{value:2}" for value in row.tolist()])
        print(row_str)

# The neural network agent will have to initialize this stuff too, and handle the loop.
if __name__ == '__main__':
    board_size = 8
    snake = t.zeros((board_size, board_size), dtype=t.int)
    #center = board_size // 2
    snake[0, :3] = T([1, 2,-1])
    score = do(snake, 1)  # needed so snake becomes 3 long and first random food spawns
    print()
    print_state(snake)
    print()

    while score != -1:
        action = input("Enter action (0: left, 1: forward, 2: right): ")
        print()
        score = do(snake, int(action) if action != '' else 1)
        print_state(snake)
        print(score)

    print('Score:', snake.max().item()-3)