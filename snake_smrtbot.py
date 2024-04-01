#!/usr/bin/env python3
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel
from time import sleep
import pygame as pg

# This is a simplified version of the classic Snake game, reworked to play
# itself without a neural network, just a old-fashion bot, with depth-first search!
# Highest 10x10 score I've seen it score thus far: 63
# Built using MiniSnakes - https://github.com/eliasffyksen/MiniSnakes

game_size = 10

pg.init()

# Set up some constants
WIDTH, HEIGHT = 500, 500  # Window size
CELL_SIZE = WIDTH // game_size  # Size of a cell in the grid

# Create the window
window = pg.display.set_mode((WIDTH, HEIGHT))

def do(snake: t.Tensor, action: int):
    prevsegs = snake.max().item()
    distb4 = getdists(snake)
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in positions]
    rotation = T([[0, -1], [1, 0]]).matrix_power(3 + action)
    pos_next = (pos_cur + (pos_cur - pos_prev) @ rotation) % T(snake.shape)
    
    if (snake[tuple(pos_next)] > 0).any():
        return -100
    
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

def draw_state(snake):
    for e in pg.event.get():
        if e.type == pg.QUIT or e.type == pg.KEYDOWN and (e.key == pg.K_ESCAPE or e.key == pg.K_q):
            pg.quit()

    window.fill((255, 255, 255))
    max_val = snake.max().item() + 1
    
    for y in range(game_size):
        for x in range(game_size):
            value = snake[y, x].item()
            if value > 0:  # Snake body
                color = (200 * (max_val - value) // max_val, 255, 200 * (max_val - value) // max_val)
            elif value == -1:  # Food
                color = (255, 0, 0)
            else:  # Empty space
                color = (255, 255, 255)
            pg.draw.rect(window, color, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pg.display.update()

def explore_path(snake, depth=0, max_depth=50):
    result = None
    futures = [snake.clone() for _ in range(3)]
    scores = [do(future, i) for i, future in enumerate(futures)]
    bestaction = scores.index(max(scores))
    bestsnake = futures[scores.index(max(scores))]
    
    if max(scores) == 10 or depth >= max_depth: # if food or max depth reached
        return bestaction
    if max(scores) != -100: # if not trapped
        result = explore_path(bestsnake, depth + 1, max_depth)
        if result == None and depth != 0: # meaning got trapped in future
            return None
    if max(scores) == -100 and depth != 0: # if trapped now, and not at depth 0
        return None
    elif depth == 0 and result == None: # if path leads to trap, try next best
        nextaction = scores.index(max(scores, key=lambda x: x != max(scores)))
        bestaction = nextaction if scores[nextaction] != -100 else bestaction
    
    return bestaction

def single_bot_game(size=10, highscore=0):
    snake = t.zeros((size, size), dtype=t.int)
    snake[0, :3] = T([1, 2,-1])
    reward = do(snake, 1)  # snake needs to grab first food so random food spawns
    #print_state(snake)
    draw_state(snake)
    print(f"{reward:<7}{snake.max().item()-3:^7}{highscore:>7}")

    while reward != -100:
        sleep(0.1)
        #futures = [snake.clone() for _ in range(3)]
        #scores = [do(future, i) for i, future in enumerate(futures)]
        #snake = futures[scores.index(max(scores))]
        best_action = explore_path(snake)
        print(best_action)
        reward = do(snake, best_action) if best_action != None else -100
        #print_state(snake)
        draw_state(snake)
        print(f"{reward:<7}{snake.max().item()-3:^7}{highscore:>7}")
        
    return snake.max().item()-3

if __name__ == '__main__':
    highscore = 0
    while highscore < game_size**2 - 3:
        highscore = max(highscore, single_bot_game(game_size, highscore))