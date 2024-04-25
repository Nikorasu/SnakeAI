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
ham = 0 #[1,1,1,2,0,1,1,1] #[1,1,1,1,2,2,1,1,1,1,1,1,0,0,1,1] #for Hamiltonian

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
        state = snake.clone()
        action = input("Enter action (0: left, 1: forward, 2: right): ")
        if len(game_data) and (action == 'b' or action == '5'):
            snake, action, reward = game_data.pop()
            turns -= 1
        elif action in ['0','1','2','']:
            action = int(action) if action != '' else ham[turns%len(ham)] if ham else 1
            reward = do(snake, action)
            if reward != -10:
                game_data.append([state, action, reward])
                turns += 1
        print_state(snake)
        print(turns, '\b\x1b[31m' if reward == -10 else '\b\x1b[32m' if reward > 9 else '\b', reward, action, sep='\t', end='\x1b[0m\n')
    
    if snake.max().item() >= game_size**2/2:
        save = input("Save game? (y/n): ")
        if save != 'n':
            num_collected += 1
            t.save(game_data, f'{folder}/game_{num_collected}.pt')
            open(f'{folder}/last', 'w').write(str(num_collected))

def group_files():
    num = int(open(f'{folder}/last', 'r').read())
    dataset = []
    for i in range(1, num + 1):
        gamedata = t.load(f'{folder}/game_{i}.pt')
        dataset += gamedata
    t.save(dataset, f'{num}pgames.pt')

if __name__ == '__main__':
    single_game()
    #group_files()