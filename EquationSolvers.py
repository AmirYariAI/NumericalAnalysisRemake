from math import factorial

from conf import *
from typing import  List


def table_header(n:int):
    word_space = (MAX_DIGITS +5 )
    row_space = n * (word_space + 1) + 12
    print("-( Gauss–Seidel Table )- ".center(row_space))
    print("iteration / X ",end="|")
    for i in range(n):
        print(f"X{i}".center(word_space),end='|')

    print(' mean error ')

def tabel_row(i:int,x_values : List[float] , error : float):

    print(f" iteration {i}  ", end="|")
    n = len(x_values)
    word_space = (MAX_DIGITS + 5)
    for i in range(n):
        print(f"{x_values[i]}".center(word_space),end='|')

    print("",error)

def gauss_seidel(a:List[List[float]] , starter_x : List[float] , b : List[float],debug_status:str = "auto") -> List[float]:

    match debug_status.lower():
        case "auto":
            debug = DEBUG
        case "true" | "on" | "1":
            debug = True
        case "false" | "off" | "0":
            debug = False
        case _:
            raise ValueError(f"{debug_status=} is invalid")

    n = len(starter_x)

    if len(a) != n or len(a[0]) != n or len(b) != n:
        raise ValueError("input in invalid")

    x : List[float] = starter_x.copy()

    if DEBUG:
        table_header(n)

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

    while error > epsilon:

        if DEBUG :
            tabel_row(i=iteration,x_values=x,error=error)

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

    if DEBUG:
        tabel_row(i=iteration, x_values=x, error=error)

    return x


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