import time
from termcolor import colored


def get_clues_from_file(file_name):
    with open(file_name, "r") as f:
        for i, line in enumerate(f):
            if i == 2:
                row_clues = []
                x = line.split("\n")[0][2:-2].split("], [")
                for row in x:
                    y = row.split(", ")
                    row_clues.append([int(elem) for elem in y])
            if i == 1:
                col_clues = []
                x = line.split("\n")[0][2:-2].split("], [")
                for col in x:
                    y = col.split(", ")
                    col_clues.append([int(elem) for elem in y])
        return row_clues, col_clues


def possibility_finder(n, clues):
    def iterative_finder(lst, clues, n):
        """lst is a list of tuples - (acc, l, curr_clue, c, b)
                    acc is a growing list of table values of length <= table.size - ex: [1,-1,1]
                    l = table_size-len(acc)
                    curr_clue = the value remaining to be used of the current clue.
                        ex: clues = [3,1], c = 0, acc = [1]
                            This would give us curr_clue = 2, since we already have an x, we only have 2 more x's to give
                            for clues[0] = 3.
                    c = the index in clues of the curr_clue
                    b = block size, with block = sum(clues) + len(clues) - 1. This is the minimum size that the remaining clues
                        have to take up. ex. [1,1,1] block size = 5

                    The algorithm iterates through each index in a line to make possibilities.
                    Ex: n = 5, clues = [1,1]
                        1st step:
                            -> add_to_temp = ([1]
                            -> [-1]
                """
        temp = []
        answers = []
        assert len(lst) > 0
        for tup in lst:
            acc, l, curr_clue, c, b = tup
            if l == 0 and b == 0:  # we have reached the end of the line and have no clues remaining
                answers.append(acc)
            elif b > l:  # we have more clues to place than we have spaces left to place them in. Skip.
                continue
            elif curr_clue == 0 and c != len(clues):  # we have reached the end of a clue and must place a space
                l -= 1
                c += 1
                if c != len(clues):
                    curr_clue = clues[c]
                    b -= 1
                add_to_temp = (acc + [-1], l, curr_clue, c, b)
                temp.append(add_to_temp)
            else:
                l -= 1
                # add -1
                if c == len(clues) or curr_clue == clues[c]:
                    # if all clues have been placed or we have not started a clue yet, we can place a -1
                    add_to_temp = (acc + [-1], l, curr_clue, c, b)
                    temp.append(add_to_temp)

                # add 1
                # since any required spaces have been taken care of by the last elif, you can always place a 1 here
                curr_clue -= 1
                b -= 1
                add_to_temp = (acc + [1], l, curr_clue, c, b)
                temp.append(add_to_temp)

        if answers:
            return answers
        else:
            return iterative_finder(temp, clues, n)

    block = sum(clues) + len(clues) - 1
    i = 0
    c = 0
    if n == block:
        temp = []
        while i < n and c < len(clues):
            cur_clue = clues[c]
            while cur_clue > 0:
                temp.append(1)
                i += 1
                cur_clue -= 1
            c += 1
            if c == len(clues):
                break
            temp.append(-1)
            i += 1
        return [temp]
    elif block > n:
        return False
    else:
        t = ([], n, clues[0], 0, block)
        return iterative_finder([t], clues, n)


def possibility_finder_given_line(line, clues):
    """Takes in a line and clues for that line. Returns every possibility for that line."""
    def iterative_finder(lst, clues, n):
        """lst is a list of tuples - (acc, l, curr_clue, c, b)
            acc is a growing list of table values of length <= table.size - ex: [1,-1,1]
            l = table_size-len(acc)
            curr_clue = the value remaining to be used of the current clue.
                ex: clues = [3,1], c = 0, acc = [1]
                    This would give us curr_clue = 2, since we already have an x, we only have 2 more x's to give
                    for clues[0] = 3.
            c = the index in clues of the curr_clue
            b = block size, with block = sum(clues) + len(clues) - 1. This is the minimum size that the remaining clues
                have to take up. ex. [1,1,1] block size = 5

            The algorithm iterates through each index in a line to make possibilities.
            Ex: line = [
        """
        temp = []
        answers = []
        assert len(lst) > 0
        for tup in lst:
            acc, l, curr_clue, c, b = tup
            if l == 0 and b == 0:  # we have reached the end of the line and have no clues remaining
                answers.append(acc)
            elif l == 0:  # we have reached the end of the line and still have clues left, so skip it
                continue
            elif b > l:  # we have more clues to place than we have spaces left to place them in. Skip
                continue
            elif curr_clue == 0 and c != len(clues): # we have reached the end of a clue and must place a space
                if line[n - l] == 1:  # there is a 1 here already. This is a contradiction. Skip.
                    continue
                else:  # place the space as usual
                    l -= 1
                    c += 1
                    if c != len(clues):
                        curr_clue = clues[c]
                        b -= 1
                    add_to_temp = (acc + [-1], l, curr_clue, c, b)
                    temp.append(add_to_temp)
            else:
                # add -1
                if line[n - l] != 1 and (c == len(clues) or curr_clue == clues[c]):
                    # if all clues have been placed or we have not started a clue yet,
                    # and there is not already a 1 there, we can place a -1
                    l -= 1
                    add_to_temp = (acc + [-1], l, curr_clue, c, b)
                    temp.append(add_to_temp)
                    l += 1
                elif line[n - l] == 1 and (c == len(clues) or curr_clue == clues[c]):
                    # we cannot place a -1 in a space where there is a 1. Skip it.
                    pass
                if line[n - l] != -1:  # if there's not a -1 there already, place a 1.
                    # add 1
                    l -= 1
                    curr_clue -= 1
                    b -= 1
                    add_to_temp = (acc + [1], l, curr_clue, c, b)
                    temp.append(add_to_temp)

        if answers:
            return answers
        else:
            return iterative_finder(temp, clues, n)

    n = len(line)
    block = sum(clues) + len(clues) - 1
    t = ([], n, clues[0], 0, block)
    return iterative_finder([t], clues, n)


# one function to check for "sure things" at the start of the puzzle
def sure_thing_finder(possibilities):
    """Checks for common values within all possibilities.
    ex: possibilities = [[-1,1,1], [1,1,-1]]
        returns x=[1], x_blank = []
        because line[1]=1 for all possibilities"""
    if not possibilities:
        return [], []
    temp = possibilities[0].copy()
    for p in possibilities[1:]:
        for i, elem in enumerate(p):
            if temp[i] != elem:
                temp[i] = 0
    x, x_blank = [], []
    for i, elem in enumerate(temp):
        if elem == 1:
            x.append(i)
        elif elem == -1:
            x_blank.append(i)
    return x, x_blank


# one function to get rid of options that won't work based off of values in line
def get_rid_of_bad_possibilities(line, possibilities):
    """Analyzes which possibilities in possibilities can work with the line.
        Returns only those possibilities that work."""

    def possibility_can_work(possibility):
        assert len(line) == len(possibility)
        for ix, elem in enumerate(line):
            if elem != 0:
                if elem != possibility[ix]:
                    return False
        return True

    ans = []
    for p in possibilities:
        if possibility_can_work(p):
            ans.append(p)
    return ans


class Picross:
    def __init__(self, n, row_clues, col_clues):
        assert len(row_clues) == len(col_clues) == n
        self.n = len(row_clues)
        self.array = [[0 for i in range(n)] for i in range(n)]
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.row_complete = [False] * self.n
        self.col_complete = [False] * self.n
        self.row_num_filled = [0] * self.n
        self.col_num_filled = [0] * self.n
        self.row_possibilities = [[]] * self.n
        self.col_possibilities = [[]] * self.n

    def total_filled(self):
        return sum(self.row_num_filled)

    def print(self):
        for ixr, row in enumerate(self.array):
            if ixr > 0 and ixr % 5 == 0:
                print("--" * (self.n + self.n // 5 - 1))
            s = ""
            for ix, elem in enumerate(row):
                if ix > 0 and ix % 5 == 0:
                    s += "|  "
                if elem == 0:
                    s += "_  "
                elif elem == -1:
                    s += colored("o  ", 'yellow')
                elif elem == 1:
                    s += colored("x  ", 'blue')
            s += "  " + str(self.row_clues[ixr]) + " " + str(sum(self.row_clues[ixr]) + len(self.row_clues[ixr]) - 1)
            print(s)
        print("--" * (self.n + self.n // 5 - 1))

        mx = 1
        for clue in self.col_clues:
            if len(clue) > mx:
                mx = len(clue)
        for i in range(mx):
            s = ""
            for ix in range(self.n):
                if ix > 0 and ix % 5 == 0:
                    s += "|  "
                if len(self.col_clues[ix]) >= i + 1:
                    x = str(self.col_clues[ix][i])
                    s += x + " " * (3 - len(x))
                else:
                    s += "   "
            print(s)
        print()

    def black(self, i, j):
        assert self.array[i][j] != -1
        if self.array[i][j] == 0:
            self.array[i][j] = 1
            self.row_num_filled[i] += 1
            self.col_num_filled[j] += 1

    def white(self, i, j):
        assert self.array[i][j] != 1
        if self.array[i][j] == 0:
            self.array[i][j] = -1
            self.row_num_filled[i] += 1
            self.col_num_filled[j] += 1

    def get_row(self, i):
        return self.array[i]

    def get_col(self, j):
        arr = []
        for row in self.array:
            arr.append(row[j])
        return arr

    def check_all_complete(self):
        def check_complete(line):
            for elem in line:
                if elem == 0:
                    return False
            return True
        for i in range(self.n):
            if check_complete(self.get_row(i)):
                self.row_complete[i] = True
            if check_complete(self.get_col(i)):
                self.col_complete[i] = True

    def shifted_elem_line_fill(self, clues, i, row_or_col):
        possiblities = possibility_finder(self.n, clues)
        if row_or_col == 'row':
            self.row_possibilities[i] = possiblities
        elif row_or_col == 'col':
            self.col_possibilities[i] = possiblities
        return sure_thing_finder(possiblities)

    def elem_line_fill(self, clues, i, row_or_col):
        m = self.n // 2 + 1
        min_spaces = len(clues) - 1
        block = sum(clues) + len(clues) - 1
        if len(clues) == 1:
            if clues[0] == self.n:
                return list(range(0, self.n)), []
            else:
                return False, False
        elif sum(clues) + min_spaces == self.n:
            i = 0
            j = 0
            ret = []
            blank = []
            for clue in clues:
                j = i + clue
                for x in range(i, j):
                    ret.append(x)
                if j < self.n:
                    blank.append(j)
                i = j + 1
            return ret, blank
        elif max(clues) > self.n - block:
            return self.shifted_elem_line_fill(clues, i, row_or_col)
        else:
            return False, False

    def first_pass(self):
        for i in range(self.n):
            x, x_blank = self.elem_line_fill(self.row_clues[i], i, 'row')
            if x:
                for j in x:
                    self.black(i, j)
            if x_blank:
                for j in x_blank:
                    self.white(i, j)

        for j in range(self.n):
            y, y_blank = self.elem_line_fill(self.col_clues[j], j, 'col')
            if y:
                for i in y:
                    self.black(i, j)
            if y_blank:
                for i in y_blank:
                    self.white(i, j)
        self.check_all_complete()

    def edge_fill(self, line, clues):
        ret, blank = [], []
        if line[0] == 1:
            ret += list(range(0, 0 + clues[0]))
            blank.append(0 + clues[0])
        if line[-1] == 1:
            ret += list(range(self.n - clues[-1], self.n))
            blank.append(self.n - clues[-1] - 1)
        return ret, blank

    def edge_pass(self):
        for i in range(self.n):
            if self.row_complete[i]:
                x, x_blank = False, False
            else:
                x, x_blank = self.edge_fill(self.get_row(i), self.row_clues[i])
            if x:
                for j in x:
                    self.black(i, j)
            if x_blank:
                for j in x_blank:
                    self.white(i, j)
        for j in range(self.n):
            if self.col_complete[j]:
                y, y_blank = False, False
            else:
                y, y_blank = self.edge_fill(self.get_col(j), self.col_clues[j])
            if y:
                for i in y:
                    self.black(i, j)
            if y_blank:
                for i in y_blank:
                    self.white(i, j)
        self.check_all_complete()

    def trim_possiblities(self, size):
        priority_queue = []
        for i in range(self.n):
            if self.row_num_filled[i] != self.n and self.row_num_filled != 0:
                priority_queue.append((self.row_num_filled[i], i, True))
            if self.col_num_filled[i] != self.n and self.col_num_filled != 0:
                priority_queue.append((self.col_num_filled[i], i, False))
        priority_queue.sort(reverse=True)
        window_size = min(len(priority_queue), size)
        priority_queue = priority_queue[:window_size]

        for tup in priority_queue:
            x, x_blank = [], []
            y, y_blank = [], []
            i = tup[1]
            is_row = tup[2]
            if is_row:
                if self.row_possibilities[i]:
                    prev = len(self.row_possibilities[i])
                    self.row_possibilities[i] = get_rid_of_bad_possibilities(self.get_row(i), self.row_possibilities[i])
                    if len(self.row_possibilities[i]) < prev:
                        x, x_blank = sure_thing_finder(self.row_possibilities[i])
                else:
                    self.row_possibilities[i] = possibility_finder_given_line(self.get_row(i), self.row_clues[i])
                    x, x_blank = sure_thing_finder(self.row_possibilities[i])
            else:
                if self.col_possibilities[i]:
                    prev = len(self.col_possibilities[i])
                    self.col_possibilities[i] = get_rid_of_bad_possibilities(self.get_col(i), self.col_possibilities[i])
                    if len(self.col_possibilities[i]) < prev:
                        y, y_blank = sure_thing_finder(self.col_possibilities[i])
                else:
                    self.col_possibilities[i] = possibility_finder_given_line(self.get_col(i), self.col_clues[i])
                    y, y_blank = sure_thing_finder(self.col_possibilities[i])
            if x:
                for j in x:
                    self.black(i, j)
            if x_blank:
                for j in x_blank:
                    self.white(i, j)
            if y:
                for j in y:
                    self.black(j, i)
            if y_blank:
                for j in y_blank:
                    self.white(j, i)

    def solve(self, print_it=False, debug=False):
        n2 = self.n * self.n
        start = time.time()
        time_so_far = start

        self.first_pass()
        self.edge_pass()
        past_total_filled = self.total_filled()
        if debug:
            self.print()

        size = self.n // 5
        while self.total_filled() != n2:
            self.trim_possiblities(size)
            if self.total_filled() == past_total_filled:
                size += self.n // 5
            self.edge_pass()
            past_total_filled = self.total_filled()
            time_so_far = time.time() - start

            if debug:
                self.print()

        if print_it:
            self.print()
        return time_so_far, self.encode_solution()

    def encode_solution(self):
        s = ""
        for row in self.array:
            for val in row:
                if val < 1:
                    s += "n"
                else:
                    s += 'y'
        return s
