from operator import methodcaller
import requests
from picross import Picross


def get_specific_picross_content(puzzle_id, size):
    """Example usage:
        content = get_specific_picross_content("4,885,225")
        row_clues, col_clues, puzzle_id, param = website_data_to_25x25_picross(content)
    """

    def id_to_num(id):
        x = id.split(',')
        n = ""
        for val in x:
            n += val
        return int(n)

    url = "https://www.puzzle-nonograms.com/"
    id = id_to_num(puzzle_id)
    post_fields = {
        'specific': 1,
        'size': size,
        'specid': id
    }
    s1 = requests.session()
    res1 = s1.post(url, data=post_fields)
    ret = str(res1.content)
    s1.close()
    return ret


def website_data_to_picross(content, size, write_file=False):
    i1 = content.find('name="param"') + 20
    i2 = content.find('"', i1)
    param = content[i1:i2]

    i1 = content.find('span id="puzzleID"') + 19
    i2 = content.find('</span>', i1)
    puzzle_id = content[i1:i2]

    i1 = content.find("var task =") + 13
    i2 = content.find(";", i1) - 2

    x = content[i1:i2]
    y = x.split('/')
    z = list(map(methodcaller("split", "."), y))
    z2 = z[:size]
    z1 = z[size:]
    row_clues = [[int(val) for val in lst] for lst in z1]
    col_clues = [[int(val) for val in lst] for lst in z2]

    if write_file:
        s = puzzle_id + ".txt"
        with open(s, "w") as f:
            f.write(puzzle_id)
            f.write("\n")
            f.write(str(row_clues))
            f.write("\n")
            f.write(str(col_clues))
    return row_clues, col_clues, puzzle_id, param


def get_and_solve_picross(size, print_puzzle=True, write_puzzle_to_file=False, post_to_site=False, account_email=None):
    assert size in [5, 10, 15, 20, 25]
    size_dict = {5: 0, 10: 1, 15: 2, 20: 3, 25: 4}

    url = "https://www.puzzle-nonograms.com/"
    if size != 5:
        url += "?size=" + str(size_dict[size])

    session = requests.session()
    res1 = session.get(url)
    row_clues, col_clues, puzzle_id, param = website_data_to_picross(str(res1.content), size, write_puzzle_to_file)
    p = Picross(len(row_clues), row_clues, col_clues)
    solve_time, solution_encoded = p.solve(print_puzzle)

    post_fields = {
        "jstimer": 0,
        # jsPersonalTimer
        "jstimerShowPersonal": 1582,
        'stopClock': 0,
        'robot': 1,
        'zoomslider': 1,
        # jstimerShow, jsShowPersonal
        'jstimerShow': "00:00",
        'jstimerShowPersonal': '00:00',
        'b': 1,
        'size': size_dict[size],
        'param': param,
        'w': 25,
        'h': 25,
        'ansH': solution_encoded,
        'ready': 'Done'
    }

    if post_to_site:
        assert account_email is not None
        res2 = session.post(url, data=post_fields)
        s2 = str(res2.content)[2:]
        s2 = s2[:-1]
        i1 = s2.find("solparam") + 18
        i2 = s2.find('/>', i1) - 2
        solparam = s2[i1:i2]

        url2 = "https://www.puzzle-nonograms.com/hallsubmit.php"
        post_fields2 = {
            "submitscore": 1,
            "solparams": solparam,
            "email": account_email
        }
        res3 = session.post(url2, data=post_fields2)
    session.close()
    return solve_time


def get_and_solve_specific_picross(puzzle_id, size, print_puzzle=True, debug_puzzle=False):
    assert size in [5, 10, 15, 20, 25]
    size_dict = {5: 0, 10: 1, 15: 2, 20: 3, 25: 4}
    content = get_specific_picross_content(puzzle_id, size_dict[size])
    row_clues, col_clues, puzzle_id, param = website_data_to_picross(content, size)
    p = Picross(len(row_clues), row_clues, col_clues)
    solve_time, solution_encoded = p.solve(print_puzzle, debug_puzzle)


def test_n_puzzles(n, size):
    tms = []
    for i in range(n):
        print(i)
        tm = get_and_solve_picross(size, print_puzzle=True)
        tms.append(tm)
    print(sum(tms)/n)


