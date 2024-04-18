#!/usr/bin/env python3
from snake import do, print_state
import torch as t
from torch import tensor as T
from time import sleep

# This version will be designed to collect data on high scoring games, in a format easy to feed into a neural network.
# Built using MiniSnakes - https://github.com/eliasffyksen/MiniSnakes

slowmode = False   # slows things down so you can watch what's going on
num2save = 10000    # number of high-scoring games to save (actual turn count may vary)
maxgames = 50000   # maximum number of games to play before giving up
game_size = 8      # has to be same size as version NN plays
threshold = 42     # threshold over which to save games, locked in this version
trimstart = True   # removes the first few moves, to help randomize the start a little
trimend = True     # removes data after 2nd to last food eaten, as it usually leads to dead-ends
game2file = True   # whether to save each game to it's own file, or to just append to memory list

def explore_path(snake, depth=0, max_depth=game_size**2//2): # hs 56 = 60/64
    futures = [snake.clone() for _ in range(3)]
    scores = [do(future, i) for i, future in enumerate(futures)]
    bestaction = scores.index(max(scores))
    bestsnake = futures[bestaction]
    if depth >= max_depth or max(scores) >= 10: return bestaction
    result = explore_path(bestsnake, depth + 1, max_depth) if max(scores) != -10 else None
    if depth == 0 and result == None: # if path leads to trap, try next best
        nextaction = scores.index(max(scores, key=lambda x: x != max(scores)))
        bestaction = nextaction if scores[nextaction] != -10 else bestaction
    return bestaction if depth == 0 else result

def trimdeath(game_data):
    tens = 0  # this function tries to remove actions that lead up to death
    for i in range(len(game_data)-1, -1, -1):
        if game_data[i][2] >= 10:
            tens += 1
            if tens == 2:
                return game_data[:i+1]
    return game_data

class GameRecorder:
    def __init__(self):
        self.cycles = maxgames
        self.bestgames_cache = []
        try: self.games_collected = int(open('bestgames/lastcount', 'r').read())
        except FileNotFoundError: self.games_collected = 0
        self.turnspergame = []
        self.scores = []
        self.highscore = 0

    def single_bot_game(self):
        game_data = []
        turns = 0
        snake = t.zeros((game_size, game_size), dtype=t.int)
        snake[0, :4] = T([1,2,3, -1])
        reward = do(snake, 1) # snake needs to grab first food so random food spawns

        while reward != -10:
            turns += 1
            state = snake.clone()
            best_action = explore_path(snake, max_depth=game_size**2-snake.max().item())
            reward = do(snake, best_action) if best_action != None else -10
            if slowmode:
                sleep(0.2)
                print_state(snake)
                print(f"{reward:<6}{snake.max().item():^6}{self.highscore:>6}{1+maxgames-self.cycles:>9}")
            if turns > 3 or not trimstart: # avoids saving some of the very first moves, to randomize start more
                game_data.append([state, best_action, reward]) # state, action, reward, next_state #, snake.clone() for next_state
                #We don't really need to save the next state, since we can get it from the following index later.
        
        print_state(snake)
        print(f"{snake.max().item():<6}{self.highscore:^6}{1+maxgames-self.cycles:>9}")
        
        if snake.max().item() >= threshold:
            self.games_collected += 1
            if game2file: t.save(game_data, f'bestgames/game_{self.games_collected}.pt')
            else:
                if trimend: game_data = trimdeath(game_data)
                self.bestgames_cache.extend(game_data)
            self.scores.append(snake.max().item())
            self.turnspergame.append(turns)
            print(f"Scored over {threshold}!  Saved: {self.games_collected}")
        
        return snake.max().item()
    
    def run(self):
        try:
            while self.games_collected < num2save and self.cycles > 0:
                self.highscore = max(self.highscore, self.single_bot_game())
                self.cycles -= 1
        except KeyboardInterrupt:
            print("\nEnding early..")
        print("\nDone!")
        print(f"Games collected:    {self.games_collected:>9}")
        print(f"Average Turns/game: {sum(self.turnspergame) / len(self.turnspergame):>9.1f}")
        print(f"Total turns saved:  {len(self.bestgames_cache):>9,}")
        print(f"Average score:      {sum(self.scores) / len(self.scores):>9.3f}")
        print(f"Highest score:      {self.highscore:>9}")
        print("\nSaving to file..")
        datafile = f"data_{'t' if trimstart else ''}{self.games_collected}{'t' if trimend else ''}_{sum(self.scores) // len(self.scores)}.pt"
        if game2file:
            dataset = []
            open('bestgames/lastcount', 'w').write(str(self.games_collected))
            for i in range(1, self.games_collected + 1):
                gamedata = t.load(f'bestgames/game_{i}.pt')
                if trimend: gamedata = trimdeath(gamedata)
                dataset.extend(gamedata)
            t.save(dataset, datafile)
        else:
            t.save(self.bestgames_cache, datafile)

if __name__ == '__main__':
    tutor = GameRecorder()
    tutor.run()