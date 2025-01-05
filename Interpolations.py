from typing import List, Any, Callable
from conf import *
import time

def nameof(var:Any):
    return f'{var=}'.split('=')[0]

class BasicInterpolation:

    F_points: List[float] = []
    X_points: List[float] = []
    MinX: float = None
    MaxX: float = None

    def __init__(self, x_points:List[float], f_points:List[float], process_point:bool=True) -> None:

        n : int = len(x_points)

        if n < 2 :
            raise ValueError("this program need two points at least.")

        if n != len(f_points) :
            raise ValueError(f"{nameof(x_points)} and {nameof(f_points)} must be at the same length.")

        if process_point:
            self.F_points = f_points.copy()
            self.X_points = x_points.copy()

            self.MinX = x_points[0]
            self.MaxX = x_points[0]

            for index in range(1,n):
                self.MinX = min(x_points[index],self.MinX)
                self.MaxX = max(x_points[index],self.MaxX)

        if DEBUG:
            print("init executed")

    def __call__(self,x:float) -> float:
        return  self.predict(x)

    def predict(self,x: float) -> float:

        if not(self.MinX <= x <= self.MaxX):
            raise ValueError(f"{x=} must be in the range of [{self.MinX} , {self.MaxX}]")

        return 0.0


    def __str__(self) -> str:
        return "Basic Interpolation"

    def __repr__(self) -> str:
        return "BasicInterpolation()"

    def add_point(self,x:float,f:float) -> None:

        if x in self.X_points:
            raise  ValueError(f"{x=} is already in the list")

    def save(self,path:str) -> bool:
        pass

    def load(self,path:str) -> bool:
        pass

class Lagrange(BasicInterpolation):

    def __init__(self,x_points:List[float],f_points:List[float]):
        super().__init__(x_points = x_points,f_points = f_points,process_point=True)

    def __lagrange_multiplier(self,x:float,index :int) -> float:

        xi : float = self.X_points[index]
        l : float = 1.0

        for j,xj in enumerate(self.X_points):
            if index == j : continue
            l *= round((x - xj) / (xi - xj),MAX_DIGITS)

        return round(l,MAX_DIGITS)

    def predict(self,x : float) -> float:

        if not(self.MinX <= x <= self.MaxX):
            raise ValueError(f"{x=} must be in the range of [{self.MinX} , {self.MaxX}]")

        fx : float =  0.0
        start : float = 0

        if DEBUG:
            start = time.time()
            print(f"->> Lagrange Interpolation for f({x}):")

        for i,fi in enumerate(self.F_points):

            l = self.__lagrange_multiplier(x,i)
            fx += round(l * fi,MAX_DIGITS)

            if DEBUG:
                print(f'({l})*{fi} ',end="+ ")

        fx = round(fx,MAX_DIGITS)

        if DEBUG:
            print(f"\b\b= {fx}")
            execution_time = time.time() - start
            print(f"Execution Time : {execution_time} s")

        return fx

    def add_point(self,x:float,f:float) -> Callable[[float],float] :

        super().add_point(x,f)

        x_points = self.X_points.copy()
        f_points = self.F_points.copy()

        x_points.extend([x])
        f_points.extend([f])

        new_obj = Lagrange(x_points,f_points)
        return new_obj

class DiffError(Exception):
    pass

class Newton(BasicInterpolation):

    class FiniteDiff(BasicInterpolation):

        h : float = None
        def __init__(self,x_points:List[float],f_points:List[float]):
            super().__init__(x_points = x_points,f_points = f_points,process_point=False)

            temp_x = [(x,f) for x,f in zip(x_points,f_points)]
            temp_x.sort(key = lambda point : point[0])

            self.X_points = []
            self.F_points = []
            for x,f in temp_x:
                self.X_points.append(x)
                self.F_points.append(f)

            self.MinX = self.X_points[0]
            self.MaxX = self.X_points[-1]

            px = self.X_points[0]
            for x in self.X_points[1:]:
                if not self.h:
                      self.h = x - px
                elif self.h != (x - px):
                    raise DiffError(f"{nameof(x_points)} Must have a same differences")
                px = x

            print("FiniteDiff")

    class DividedDifferences(BasicInterpolation):

        def __init__(self, x_points: List[float], f_points: List[float]):
            super().__init__(x_points=x_points, f_points=f_points,process_point=True)
            print("FiniteDiff")

    def __init__(self,x_points:List[float],f_points:List[float]):

        super().__init__(x_points=x_points,f_points=f_points,process_point=False)

        try:
            self.__interpolation = Newton.FiniteDiff(x_points,f_points)
        except DiffError:
            self.__interpolation = Newton.DividedDifferences(x_points,f_points)
        except Exception as e:
            raise e

    def __repr__(self):
        return repr(self.__interpolation)
    def __str__(self):
        return  str(self.__interpolation)
    def __call__(self,x:float):
        return self.__interpolation(x)

    def save(self,path:str) -> bool:
        return self.__interpolation.save(path)

    def load(self,path:str) -> bool:
        return self.__interpolation.load(path)

    def add_point(self,x:float,f:float):
        return self.__interpolation.add_point(x,f)

if __name__ == "__main__":

    print("Hello World")
    x2 = Lagrange([1,2], [20,10])
    print(x2(1.5))

    # inp = input()
    # X , Y = [] , []
    # while inp != "":
    #    x , y = map(float,inp.split())
    #    X.append(x)
    #    Y.append(y)
    #    if len(X) > 1 :
    #        print(x - X[-2])
    #    inp = input()

    x3 = x2.add_point(1.5,0)
    print(x3(1.5))
