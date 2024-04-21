#!/usr/bin/env python3
from snake import do, print_state
import torch as t
from torch import tensor as T

# This version allows recording a manually played game of Snake step-by-step,
# in order to record a few perfect games with better moves than the bot can.
# Also allows for "backspacing" moves when needed by entering 'b', otherwise
# the player enters the action number (0: left, 1: forward, 2: right).
# Enter 'q' to quit prematurely. (hitting Enter with no input counts as 1)

game_size = 8      # has to be same size as version NN plays
folder = 'mygames'

def single_game():
    try: num_collected = int(open(f'{folder}/last', 'r').read())
    except FileNotFoundError: num_collected = 0
    game_data = []
    turns = 0
    snake = t.zeros((game_size, game_size), dtype=t.int)
    snake[0, :4] = T([1,2,3, -1])
    reward = do(snake, 1) # snake needs to grab first food so random food spawns
    action = ''
    print_state(snake)
    print()

    while action != 'q' and snake.max().item() < game_size**2:
        turns += 1
        state = snake.clone()
        action = input("Enter action (0: left, 1: forward, 2: right): ")
        
        if action == 'b':
            turns -= 2
            snake, action, reward = game_data.pop()
        elif action in ['0','1','2','']:
            action = int(action) if action != '' else 1
            reward = do(snake, action)
            game_data.append([state, action, reward]) # state, action, reward
        
        print_state(snake)
        print(f"{turns:<4}{reward:^5}{action:^5}{snake.max().item():^5}")
    
    if snake.max().item() >= game_size**2/2:
        save = input("Save game? (y/n): ")
        if save != 'n':
            num_collected += 1
            t.save(game_data, f'{folder}/game_{num_collected}.pt')
            open(f'{folder}/last', 'w').write(str(num_collected))

if __name__ == '__main__':
    single_game()