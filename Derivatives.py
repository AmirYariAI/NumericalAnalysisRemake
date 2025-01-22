import math
from typing import List
from conf import *
from DebugAssistant import debug_status
from Interpolations import Newton

class NewtonDerivative(Newton.FiniteDifferences):

    def __init__(self, x_points: List[float], f_points: List[float]) -> None:
        super().__init__(x_points=x_points,f_points=f_points,is_forward=True,debug_mode="0")

    def __call__(self,x:float ,step:int=-1,debug_mode:str = "auto") -> float:
        return  self.predict(x=x,step=step,debug_mode = debug_mode)

    def predict(self,x: float,step:int=-1,debug_mode:str = "auto") -> float:
        if x not in self.X_points:
            raise ValueError(f"{x=} is invalid")

        return self.__predict_derivative(x = x , s = 0 , step = step,debug_mode=debug_mode)

    def predict_next_step(self, x:float,step:int=-1,debug_mode:str = "auto") -> float:

        if x not in self.X_points:
            raise ValueError(f"{x=} is invalid")

        return self.__predict_derivative(x=x, s=1,step=step,debug_mode=debug_mode)

    def predict_next_half_step(self, x:float,step:int=-1,debug_mode:str = "auto") -> float:

        if x not in self.X_points:
            raise ValueError(f"{x=} is invalid")

        return self.__predict_derivative(x=x, s=0.5,step=step,debug_mode=debug_mode)

    def __predict_derivative(self, x : float, s : float, step:int=-1,debug_mode:str = "auto") -> float:

        debug = debug_status(debug_mode)

        x_index = self.X_points.index(x)
        result = 0
        n = min(len(self.diff_table) - x_index , step+1) if step != -1 else len(self.diff_table) - x_index

        if debug:
            print(f"f'({x}) =(", end='')

        for k in range(1, n):

            factor = self.__s_factor(s = s,k = k)
            result += round(self.diff_table[k][x_index] * factor, MAX_DIGITS)

            if debug:
                print(f" {self.diff_table[k][x_index]} * ({factor}) ", end='+')

        result = round(result * (1 / self.h),MAX_DIGITS)

        if debug:
            print(f"\b) * ({1/self.h})= {result} (step : {n - 1})")
            self.__print_table(x_index=x_index,step=step)

        return result

    @staticmethod
    def __s_factor(s:float,k:int) -> float:

        factor = 0

        for i in range(k):
            d = 1

            for j in range(k):
                if i == j:
                    continue
                d = round(d *(s - j),MAX_DIGITS)

            factor += d

        for i in range(1 , k + 1):
            factor = round(factor / i , MAX_DIGITS)

        return factor

    def __print_table(self,x_index = -1,step:int = -1):

        n = len(self.F_points)
        selector_open, selector_close = "(", ")"

        word_space = math.ceil(math.log10(self.MaxX)) + 2 + MAX_DIGITS
        print("-( Finite Differences Table )-".center(13 + n * word_space + n + 1))
        print("Δ^i f(x)\\ Xi ", end="|")
        for i in range(n):
            print(f"{self.X_points[i]}".center(word_space), end='|')
        print()

        print(f"f(x)".center(13), end="|")

        for index,f in enumerate(self.diff_table[0]):
            print(f"{f}".center(word_space), end="↓")

        print("\b")

        for i in range(1, n):

            print(f"Δ^{i} f(x)".center(13), end="|")

            print(" " * ((word_space // 2 + 1) * i), end="")

            for index,f in enumerate(self.diff_table[i]):

                if step == -1 or i <= step :
                    if index == x_index <= n - (i+1)  :
                        point_str = f"{selector_open}{f}{selector_close}"
                        print(point_str.center(word_space), end="↓")
                        continue

                print(f"{f}".center(word_space), end="↓")

            print("\b")

def sc(n):
    x11 , f11 = [] , []
    for i in range(n):
        x11.append(i)
        f11.append(i**2)
    return x11,f11

if __name__ == "__main__":

    print("Test 1")
    t1_x_points, t1_f_points = [0.1,0.2,0.3,0.4],[-1,2,3,5]
    a = NewtonDerivative(t1_x_points,t1_f_points)
    a.predict(x=0.1,step = -1)



