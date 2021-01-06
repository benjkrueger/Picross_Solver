# Picross_Solver
This program grabs a Nonogram puzzle from https://www.puzzle-nonograms.com/ and solves it.

## Two files:
### picross.py: contains all the logic for setting up and solving a picross puzzle.
#### Relevant functions:
##### Picross(row_clues, col_clues)
* row_clues and col_clues are both lists of lists of integers. Ex: [3,5]
### url_func.py: contains all of the logic for getting puzzles from https://www.puzzle-nonograms.com/ and solving them.
#### Relevant functions:
##### get_and_solve_picross(size, print_puzzle=True, write_puzzle_to_file=False, post_to_site=False, account_email=None)
* size is one of [5,10,15,20,25]
* print_puzzle decides whether a picture of the solved puzzle is printed to console
* write_puzzle_to_file decides whether you want to create a text document that writes down the clues and puzzle id of the puzzle
* post_to_site enables the posting of the results to the website for high scores
* account_email is needed if post_to_site is enabled
##### get_and_solve_specific_picross(puzzle_id, size, print_puzzle=True, debug_puzzle=False)
* puzzle_id is the unique puzzle id of the puzzle you are looking for
* size is one of [5,10,15,20,25]
* print_puzzle decides whether a picture of the solved puzzle is printed to console
* debug_puzzle decides whether you want to print a picture of the puzzle after each iteration of the solving algorithm
##### test_n_puzzles(n, size)
* n is the number of times you want to test
* size is one of [5,10,15,20,25]
