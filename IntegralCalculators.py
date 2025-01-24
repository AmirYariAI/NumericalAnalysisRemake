from conf import *
from typing import List
from DebugAssistant import debug_status
from Interpolations import DiffError
from EquationSolvers import EquationSolvers


class NewtonCotes:
    h: float = None

    def __init__(self,x_points:List[float],f_points:List[float]) -> None:

        n: int = len(x_points)

        if n < 2:
            raise ValueError("this program need two points at least.")

        if n != len(f_points):
            raise ValueError(
                f"{f"{x_points=}".split('=')[0]} and {f"{f_points=}".split('=')[0]} must be at the same length.")

        temp_x = [(x, f) for x, f in zip(x_points, f_points)]
        temp_x.sort(key=lambda point: point[0])

        self.X_points = []
        self.F_points = []
        for x, f in temp_x:
            self.X_points.append(x)
            self.F_points.append(f)

        self.MinX = self.X_points[0]
        self.MaxX = self.X_points[-1]

        px = self.X_points[0]
        for index,x in enumerate(self.X_points[1:]):

            self.X_points[index] = round(self.X_points[index] - self.MinX)

            if not self.h:
                self.h = x - px
            elif self.h != round(x - px, MAX_DIGITS):
                raise DiffError(f"{f"{x_points=}".split('=')[0]} Must have a same differences")
            px = x

        self.X_points[-1] = round(self.X_points[-1] - self.MinX)
        self.__build()

    def __build(self):

        n = len(self.X_points)

        nh = round((n-1) * self.h,MAX_DIGITS)

        b = [round(round(nh ** (i + 1),MAX_DIGITS) / (i + 1),MAX_DIGITS) for i in range(n)]
        a = []
        w = [0] * n

        for i in range(n):
            row = []
            for j in range(n):
                value = round(round(j * self.h , MAX_DIGITS) ** i,MAX_DIGITS)
                row.append(value)
            a.append(row)

        self.w = EquationSolvers.gauss_seidel(a, w, b,debug_mode='off')



    def execute(self,debug_mode : str = "True") -> float:

        debug = debug_status(debug_mode)
        ans = 0

        n = len(self.X_points)
        if debug:
            print(f"integral(f(x),{self.MinX},{self.MaxX}):")

        for i in range(n):

            if debug:
                print(f"({self.w[i]}) * ({self.F_points[i]}) ",end='+ ')

            value = round(self.w[i] * self.F_points[i],MAX_DIGITS)
            ans   = round(ans + value,MAX_DIGITS)

        ans = round(ans,MAX_DIGITS)

        if debug:
            print(f"\b\b= {ans}")

        return  ans

if __name__ == "__main__":

    x_test = [5 , 6 , 7 , 8]
    f_test = [15 , 16 , 17 ,18]

    print(NewtonCotes(x_test,f_test).execute())