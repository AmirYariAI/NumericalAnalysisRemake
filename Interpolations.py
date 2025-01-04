from typing import List,Callable
from conf import *
import sys

def nameof(var):
    return f'{var=}'.split('=')[0]

class Lagrange:

    F_points : List[float] = []
    X_points: List[float] = []
    MinX : float = None
    MaxX : float = None

    def __init__(self,x_points:List[float],f_points:List[float]) -> None:

        n : int = len(x_points)

        if n < 2 :
            raise ValueError("this program need two points at least.")

        if n != len(f_points) :
            raise ValueError(f"{nameof(x_points)} and {nameof(f_points)} must be at the same length.")

        self.F_points = f_points.copy()
        self.X_points = x_points.copy()

        self.MinX = x_points[0]
        self.MaxX = x_points[0]

        for index in range(n):
            self.MinX = min(x_points[index],self.MinX)
            self.MaxX = max(x_points[index],self.MaxX)

        if DEBUG:
            print("init executed")

    def __call__(self,x:float) -> float:
        return  self.predict(x)

    def predict(self,x : float) -> float:

        if not(self.MinX <= x <= self.MaxX):
            raise ValueError(f"{x=} must be in the range of [{self.MinX} , {self.MaxX}]")

        fx : float =  0.0

        if DEBUG:
            print(f"->> Lagrange Interpolation for f({x}):")

        for i,fi in enumerate(self.F_points):

            l = self.LagrangeMultiplier(x,i)
            fx += round(l * fi,MAX_DIGITS)

            if DEBUG:
                print(f'({l})*{fi} ',end="+ ")

        fx = round(fx,MAX_DIGITS)

        if DEBUG:
            print(f"\b\b= {fx}")

        return fx

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass


    def LagrangeMultiplier(self,x,index :int) -> float:

        xi : float = self.X_points[index]
        l : float = 1.0

        for j,xj in enumerate(self.X_points):
            if index == j : continue
            l *= round((x - xj) / (xi - xj),MAX_DIGITS)

        return round(l,MAX_DIGITS)

    def add_point(self,x,f):

        if x in self.X_points:
            raise  ValueError(f"{x=} is already in the list")

        x_points = self.X_points.copy()
        f_points = self.F_points.copy()

        x_points.extend([x])
        f_points.extend([f])

        new_obj = Lagrange(x_points,f_points)
        return new_obj

    def save(self):
        pass

    def load(self):
        pass

if __name__ == "__main__":

    print("Hello World")
    x2 = Lagrange([1,2], [20,10])
    print(x2(1.5))

    x3 = x2.add_point(1.5,0)
    print(x3(1.5))
    sys.exit(0)
