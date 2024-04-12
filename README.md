# Using ML to play Snake

This project is still being worked on, and it's mostly just to satisfy my curiosity.
Using a modified version of [MiniSnakes](https://github.com/eliasffyksen/MiniSnakes),
I've setup the base game with rewards, hopefully that can work with for reinforcement learning.
Also made an old-school bot version (`snake_smrtbot.py` & `snake_tutor.py`) to record data for training.
The bot version can usually fill over half the play area with snake, highest score I've seen was 60/64!
The current reward system gives 10 points for eating food, -10 for running into itself.
As well as a small reward as it moves around, positive values when actions move it closer to food,
negative when it's actions head away from food, scaling from 1 up to 9 (spot next to food) as it gets closer.

I haven't had much luck getting reinforcement learning working yet, so I've been trying simpler methods.
Using the _tutor version, I recorded thousands of games to file, along with the action each turn made.
That sorta gives me enough data to train a basic neural network on what move to make for whatever
the game state might currently look like.
