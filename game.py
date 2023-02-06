#!/usr/bin/env python3

# Stripped-down version of my Snake game, meant for use with a neural network! (more numpy)
# This version is not meant to be played by a human, I hope to try to train an AI to play it.
# Copyright (C) 2022  Nik Stromberg - nikorasu85@gmail.com

import numpy as np

SIZE = (20, 20)

class Snake:
    def __init__(self, size=SIZE):
        self.rows, self.cols = size
        self.pos = np.array((self.rows//2-1, self.cols//2-1), int) # start position
        self.actdir = np.random.randint(0,3)
        self.dir = np.array(((-1,0),(0,1),(1,0),(0,-1))[self.actdir], int) # random direction (up, right, down, left)
        self.segments = np.array([self.pos], int) # array of previous positions
        self.len = 3 # starting length of snake
        self.food = np.array((np.random.randint(self.rows), np.random.randint(self.cols)), int) # random food position
        self.state = np.zeros((self.rows, self.cols), int)
        self.state[self.food[0], self.food[1]] = 2
        self.state[self.segments[:,0], self.segments[:,1]] = 1
        self.gameover = False
    def update(self, action=1): # left -1, straight 0, right 1 (take 0=left, 1=straight, 2=right then -1 to convert)
        self.actdir = (self.actdir + (action-1)) % 4
        self.dir = ((-1,0),(0,1),(1,0),(0,-1))[self.actdir]
        hitself = np.all(self.pos + self.dir == self.segments[1:], axis=1).any()
        if np.any(self.pos + self.dir < 0) or np.any(self.pos + self.dir >= [self.rows, self.cols]) or hitself:
            self.gameover = True; return # if snake hits border or itself, game over
        self.pos += self.dir # update position
        self.segments = np.concatenate([[self.pos], self.segments[:self.len-1]]) # add new position to segments
        if np.all(self.pos == self.food): # if snake eats food
            self.len += 1 # increase length of snake
            field = {(x, y) for x in range(self.rows) for y in range(self.cols)}
            freespace = list(field - set(map(tuple, self.segments)))
            self.food = freespace[np.random.randint(len(freespace))]
        self.state[:] = 0 # reset state
        self.state[self.food[0], self.food[1]] = 2 # update food position
        self.state[self.segments[:,0], self.segments[:,1]] = 1 # update snake positions
        return self.state

def main():
    try:
        import time
        print('\x1b[2J\x1b[H\x1b]0;Snake',end='\a',flush=True)
        player = Snake()
        print(player.state)
        while (action:=input()) != 'q': # will need to be changed to AI input later
            start = time.perf_counter()
            print('\x1b[0m\x1b[2J\x1b[H',end='',flush=True) # clear screen
            player.update(1 if action == '' else int(action)) # 0=left, 1=straight, 2=right
            print(player.state)
            end = time.perf_counter()
            howlong = end - start
            print(howlong)
            if player.gameover: break # if game over, break out of loop
    except KeyboardInterrupt: pass # catch Ctrl+C
    finally: print(f'\x1b[0mGame Over! Score: {(player.len-3)}') # reset terminal, show cursor, print score

if __name__ == '__main__':
    main() # by Nik
