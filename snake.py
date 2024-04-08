#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
from time import sleep

# This is a simplified version of the classic Snake game, reworked specifically for implementing into machine learning.
# I've added a reward system to the do() function's return. Reward increases up to 9 the closer snake gets to food, moving away returns the negative.
# When food is eaten, that will reward 10. Game over indicated by a reward of -10, score is snake.max().item()-4
# The neural network inputs will correspond directly to the snake tensor, which I think is 8 by 8, so 64 input neurons?
# The neural network outputs will correspond to the available actions (0, 1, 2) for left, forward, right.
# Based on MiniSnakes - https://github.com/eliasffyksen/MiniSnakes

manual_input = False  # for debug testing

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
    return 10+segs-4 if segs > prevsegs else (max(int(10-distaf),1) if distaf < distb4 else min(int(-(10-distaf)),-1))

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
    if not manual_input:
        from agent import Play
        play = Play('snakemodel_7000_64-256x6-64x2-3.pth') # for neural network input
    
    board_size = 8
    count = 1 if manual_input else 10
    endscores = []
    
    while count > 0:
        timeout = 42 # if snake goes 42 turns without eating end game
        snake = t.zeros((board_size, board_size), dtype=t.int)
        snake[0, :4] = T([1, 2, 3, -1]) # snake starts off 4 long (after next line), so NN learns not to crash into self early.
        reward = do(snake, 1)  # snake needs to grab first food so random food spawns
        print()
        print_state(snake)
        print()

        while reward != -10:
            if manual_input:
                action = input("Enter action (0: left, 1: forward, 2: right): ") # for manual human input
            else:
                action = play.turn(snake) # for neural network input
            print()
            reward = do(snake, int(action) if action != '' else 1)
            print_state(snake)
            timeout -= 1
            if timeout == 0:
                reward = -10
            elif reward >= 10:
                timeout = 42
            print(f"{reward}  {'Got Food!' if reward>=10 else 'Game Over!' if reward==-10 else ''}")
            sleep(0.2)

        endscores.append(snake.max().item()-4)
        print('Score:', endscores[-1])
        sleep(1)
        count -= 1
    
    print(f'\nHigh Score:     {max(endscores):>4}')
    print(f'Average score:  {sum(endscores) / len(endscores):>4.1f}')
    print(f'\nScores: {endscores}')