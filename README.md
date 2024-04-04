# Half-working..

## Training a Neural Network to play Snake

This project is still being worked on, and it's mostly just to satisfy my curiosity.
So far I have the base game setup for a NN, with rewards, built from a modified version of MiniSnakes.
Also, built a bot-version to record some decent data for use with a ReplayBuffer.
I'm still trying to find the right parameters to actually get the NN to learn to play well.

I also made the dumbot and smrtbot versions along the wya, which play themselves using old-fashion bot methods.

Currently, my version of snake rewards 10 for eating food, -10 for running into self.
As well as a small reward as it moves around, positive values when actions take it closer to food,
negative when it's actions head away from food, scaling from 1 up to 9 (spot next to food) as it gets closer.
