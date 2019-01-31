# Reversi Game Analyzer
The code in this repository allows to analyze the [Reversi (aka. Othello)](https://en.wikipedia.org/wiki/Reversi) 
board game, which is a perfect information, zero-sum, two-player strategy game played on an 8x8 board. 
The ultimate goal of this project is to find a winner in a perfectly played game (the number of possible positions 
in Reversi is estimated to be close to, though somewhat higher than that in Checkers). 

The current algorithm solves the 6x6 version of Reversi in 6-10 hours depending on the hardware configuration. It took
[Feinstein](https://www.ics.uci.edu/~eppstein/cgt/othello.html) approximately 1.5 weeks to achieve the same result. 
While I do have access to more computational power, my code seems to be more efficient than his (i.e. I had to analyse
significantly fewer moves to achieve the result). 

## The algorithm 

The current algorithm takes into account symmetries, employs multithreading, and uses minimax search to train a series
of models (as described in [Engel, 2015](https://www.cs.umd.edu/sites/default/files/scholarly_papers/Engel.pdf)) to 
evaluate the possible moves and speed up the analysis. A very simple neural-network architecture is currently used to 
train the models, but a somewhat more complex CNN architecture would probably be used in the future (as described in 
[Liskowski *et al.*, 2018](https://arxiv.org/pdf/1711.06583.pdf)) with a different machine-learning library.

Run the main in Main.java to launch the program (for 6x6 version of Reversi). The logger will report the number of
moves analysed and the branching factors. It will also print every state analysed (up
until a certain depth in a game tree, which is currently set to 13, see TRACE_LEVEL in 
StateAnalyzer.java). After completion, the winner (white) will be reported.

## Dependencies

- log4j
- weka 3.8.3 - will probably be replaced with deeplearning4j in the future
