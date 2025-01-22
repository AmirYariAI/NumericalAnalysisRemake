from conf import *
from DebugAssistant import debug_status
from typing import  List


def debug_table_header(n:int):
    word_space = (MAX_DIGITS +5 )
    row_space = 14 + n * (word_space + 1) + max(word_space,12) +  + 7

    print("-( Gauss–Seidel Table )- ".center(row_space))
    print("iteration \\ X ",end="|")
    for i in range(n):
        print(f"X{i}".center(word_space),end='|')

    print('mean error'.center(max(word_space,12)),' note ',sep='|')

def debug_tabel_row(i:int,x_values : List[float] , error : float,note:str=""):

    print(f" iteration {i}  ", end="|")
    n = len(x_values)
    word_space = (MAX_DIGITS + 5)
    for i in range(n):
        print(f"{x_values[i]}".center(word_space),end='|')

    print(f" {error}".center(max(word_space,12)) , end='|')
    print(" " + note)

def gauss_seidel(a:List[List[float]] ,
                 starter_x : List[float] ,
                 b : List[float],
                 max_iteration:int = 20,
                 debug_mode:str = "auto") -> List[float]:

    debug = debug_status(debug_mode)

    n = len(starter_x)

    if len(a) != n or len(a[0]) != n or len(b) != n:
        raise ValueError("input in invalid")

    x : List[float] = starter_x.copy()

    if debug:
        debug_table_header(n)

    epsilon = 10 ** (-MAX_DIGITS)

    error = 0
    for i in range(n):
        eq_error = 0

        for j in range(n):
            eq_error += round(a[i][j] * x[j], MAX_DIGITS)

        eq_error = round(eq_error - b[i])
        error = round(error + abs(eq_error), MAX_DIGITS)

    error = round(error / n, MAX_DIGITS)

    iteration = 0

    while error > epsilon and iteration < max_iteration:

        if debug :
            debug_tabel_row(i=iteration,x_values=x,error=error)

        for i in range(n):
            eq_value = b[i]
            for j in range(n):
                if i == j:
                    continue

                eq_value -= round(a[i][j] * x[j], MAX_DIGITS)

            factor = round(1/a[i][i],MAX_DIGITS)
            x[i] = round(factor * eq_value,MAX_DIGITS)

        error = 0
        for i in range(n):
            eq_error = 0

            for j in range(n):
                eq_error += round(a[i][j] * x[j], MAX_DIGITS)

            eq_error = round(eq_error - b[i])
            error = round(error + abs(eq_error), MAX_DIGITS)

        error = round(error / n, MAX_DIGITS)

        iteration += 1

    if debug:
        note = "" if iteration < max_iteration else "max_iteration reached"

        debug_tabel_row(i=iteration, x_values=x, error=error,note=note)

    return x


if __name__ == "__main__":

    #4x + y + 2z = 4
    #3x + 5y + z = 7
    #x + y + 3z = 3

    a = [[4 , 1 , 2],
        [3 , 5 , 1],
        [1 , 1 , 3]]

    b = [4 , 7 , 3]
    x = [0 , 0 , 0]

    x=gauss_seidel(a,x,b)
    print(x)