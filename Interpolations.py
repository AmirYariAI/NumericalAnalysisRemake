import math
from typing import List
from conf import *
import time

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
            raise ValueError(f"{f"{x_points=}".split('=')[0]} and {f"{f_points=}".split('=')[0]} must be at the same length.")

        if process_point:
            self.F_points = f_points.copy()
            self.X_points = x_points.copy()

            self.MinX = x_points[0]
            self.MaxX = x_points[0]

            for index in range(1,n):
                self.MinX = min(x_points[index],self.MinX)
                self.MaxX = max(x_points[index],self.MaxX)

        if DEBUG:
            print("Basic init executed")

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
        if DEBUG:
            print('Lagrange init executed')

    def __lagrange_multiplier(self,x:float,index :int) -> float:

        xi : float = self.X_points[index]
        l : float = 1.0

        for j,xj in enumerate(self.X_points):
            if index == j : continue
            l *= round((x - xj) / (xi - xj),MAX_DIGITS)

        return round(l,MAX_DIGITS)

    def predict(self,x : float) -> float:

        super().predict(x)

        fx : float =  0.0
        start : float = 0

        if DEBUG:
            start = time.time()
            print(f"->> Lagrange Interpolation for p({x}):")

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

    def add_point(self,x:float,f:float) -> None :

        super().add_point(x,f)

        self.X_points.append(x)
        self.F_points.append(f)

class DiffError(Exception):
    pass

class Newton(BasicInterpolation):

    class FiniteDifferences(BasicInterpolation):

        diff_table : List[List[float]] = [[]]
        h : float = None
        is_forward:bool = True

        def __init__(self,x_points:List[float],f_points:List[float],is_forward:bool = True)-> None:

            super().__init__(x_points = x_points,f_points = f_points,process_point=False)
            self.is_forward = is_forward

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
                elif self.h != round(x - px,MAX_DIGITS):
                    print(self.h , x - px)
                    raise DiffError(f"{f"{x_points=}".split('=')[0]} Must have a same differences")
                px = x

            self.diff_table = self.__create_table()
            if DEBUG:
                self.__print_table()
                print("FiniteDifferences init executed ")

        def __print_table(self):

            n = len(self.F_points)
            selector_open,selector_close = "(",")"

            word_space = math.ceil(math.log10(self.MaxX)) + 2 + MAX_DIGITS
            print("-( Finite Differences Table )-".center(13 + n * word_space + n+1))
            print("Δ^i f(x)\\ Xi ", end="|")
            for i in range(n):
                print(f"{self.X_points[i]}".center(word_space), end='|')
            print()

            print(f"f(x)".center(13), end="|")

            point_str = f"{self.diff_table[0][0]}"
            if self.is_forward:
                point_str = selector_open + point_str + selector_close
            print(point_str.center(word_space), end="↓")
            for f in self.diff_table[0][1:-1]:
                print(f"{f}".center(word_space), end="↓")

            point_str = f"{self.diff_table[0][-1]}"
            if not self.is_forward:
                point_str = selector_open + point_str + selector_close
            print(point_str.center(word_space), end="↓")

            print("\b")

            for i in range(1, n):

                print(f"Δ^{i} f(x)".center(13), end="|")

                print(" " * ((word_space // 2 + 1) * i), end="")

                point_str = f"{self.diff_table[i][0]}"
                if self.is_forward or i+1 == n:
                    point_str = selector_open + point_str + selector_close
                print(point_str.center(word_space), end="↓")

                for f in self.diff_table[i][1:-1]:
                    print(f"{f}".center(word_space), end="↓")

                if i + 1!= n:
                    point_str = f"{self.diff_table[i][-1]}"
                    if not self.is_forward:
                        point_str = selector_open + point_str + selector_close
                    print(point_str.center(word_space), end="↓")

                print("\b")

        def __create_table(self) -> List[List[float]]:
            n = len(self.F_points)

            table :List[List[float]] = [self.F_points.copy()]
            for i in range(n-1,0,-1):
                row : List[float] = []

                for j in range(i):
                    f = table[-1][j+1]
                    pf = table[-1][j]
                    diff = round(f - pf, MAX_DIGITS)
                    row.append(diff)

                table.append(row)

            return  table

        def __update_table_min(self,f:float) -> List[List[float]]:
            n = len(self.F_points)

            table = self.diff_table.copy()
            table[0].insert(0,f)

            for i in range(1,n-1):
                f = table[i - 1][1]
                pf = table[i - 1][0]
                diff = round(f - pf, MAX_DIGITS)
                table[i].insert(0,diff)

            f = table[- 1][1]
            pf = table[- 1][0]

            diff = round(f - pf, MAX_DIGITS)
            table.append([diff])

            return table

        def __update_table_max(self,f:float)->List[List[float]]:
            n = len(self.F_points)

            table = self.diff_table.copy()
            table[0].append(f)

            for i in range(1, n - 1):
                f = table[i - 1][-1]
                pf = table[i - 1][-2]
                diff = round(f - pf,MAX_DIGITS)
                table[i].append(diff)

            f = table[- 1][-1]
            pf = table[- 1][-2]
            table.append([f - pf])

            return table

        def predict(self,x: float) -> float:

            super().predict(x)
            if self.is_forward:
                result = self.__predict_forward(x)
            else:
                result = self.__predict_backward(x)

            return result

        def __predict_forward(self, x:float) -> float:
            result = self.diff_table[0][0]
            s = (x - self.MinX) / self.h
            s = round(s,MAX_DIGITS)
            newton_factor = 1
            n = len(self.diff_table)

            if DEBUG:
                print(f"p({x}) = {result} ",end='+')

            for k in range(1,n):

                factor = (s-k+1) / k
                factor = round(factor,MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][0]*newton_factor,MAX_DIGITS)

                if DEBUG:
                    print(f" {self.diff_table[k][0]} * ({newton_factor}) ", end='+')

            if DEBUG:
                print(f"\b= {result}")

            return  result

        def __predict_backward(self, x:float) -> float:
            result = self.diff_table[0][-1]
            t = (x - self.MaxX) / self.h
            t = round(t, MAX_DIGITS)
            newton_factor = 1
            n = len(self.diff_table)

            if DEBUG:
                print(f"p({x}) = {result} ",end='+')

            for k in range(1, n):
                factor = (t + k - 1) / k
                factor = round(factor, MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][-1] * newton_factor, MAX_DIGITS)

                if DEBUG:
                    print(f" {self.diff_table[k][-1]} * ({newton_factor}) ", end='+')

            if DEBUG:
                print(f"\b= {result}")

            return result

        def add_point(self,x:float,f:float) -> None:
            super().add_point(x,f)

            if x <= self.MinX :

                if self.MinX - x != self.h:
                    raise DiffError(f"{self.MinX} - {x} != {self.h}")

                self.MinX = x
                self.X_points.insert(0,x)
                self.F_points.insert(0,f)
                self.diff_table = self.__update_table_min(f)
            elif x >= self.MaxX:

                if x - self.MaxX != self.h:
                    raise DiffError(f"{x} - {self.MaxX} != {self.h}")

                self.MaxX = x
                self.X_points.append(x)
                self.F_points.append(f)
                self.diff_table = self.__update_table_max(f)
            else:
                raise DiffError(f"{x=} can not be added into the {f"{self.X_points=}".split('=')[0]}")

            if DEBUG:
                self.__print_table()

        def change_direction(self) -> None:
            self.is_forward = not self.is_forward
            if DEBUG:
                mode_str = "Forward"
                if not self.is_forward:
                    mode_str = "Backward"
                print(f"Changed to *({mode_str})* Newton Finite Differences Interpolation")

    class DividedDifferences(BasicInterpolation):

        diff_table: List[List[float]] = [[]]
        is_forward: bool = True

        def __init__(self, x_points: List[float], f_points: List[float],is_forward: bool = True)-> None:
            super().__init__(x_points=x_points, f_points=f_points,process_point=True)
            self.is_forward = is_forward
            self.diff_table = self.__create_table()
            if DEBUG:
                self.__print_table()
                print("DividedDifferences init executed ")

        def __print_table(self) -> None:

            n = len(self.F_points)
            max_n_str : int = math.floor(math.log10(n))

            selector_open, selector_close = "(", ")"

            word_space = math.ceil(math.log10(self.MaxX)) + 2 + MAX_DIGITS
            print("-( Divided Differences Table )-".center(13 + n * word_space + n + 1))
            print(" f[xi,...,xi+k] \\Xi ".center(20+max_n_str), end="|")
            for i in range(n):
                print(f"{self.X_points[i]}".center(word_space), end='|')
            print()

            print(f"f(x)".center(20+max_n_str), end="|")

            point_str = f"{self.diff_table[0][0]}"
            if self.is_forward:
                point_str = selector_open + point_str + selector_close
            print(point_str.center(word_space), end="↓")
            for f in self.diff_table[0][1:-1]:
                print(f"{f}".center(word_space), end="↓")

            point_str = f"{self.diff_table[0][-1]}"
            if not self.is_forward:
                point_str = selector_open + point_str + selector_close
            print(point_str.center(word_space), end="↓")

            print("\b")

            for i in range(1, n):

                print(f"f[xi,...xi+{i}]".center(20+max_n_str), end="|")

                print(" " * ((word_space // 2 + 1) * i), end="")

                point_str = f"{self.diff_table[i][0]}"
                if self.is_forward or i + 1 == n:
                    point_str = selector_open + point_str + selector_close
                print(point_str.center(word_space), end="↓")

                for f in self.diff_table[i][1:-1]:
                    print(f"{f}".center(word_space), end="↓")

                if i + 1 != n:
                    point_str = f"{self.diff_table[i][-1]}"
                    if not self.is_forward:
                        point_str = selector_open + point_str + selector_close
                    print(point_str.center(word_space), end="↓")

                print("\b")

        def __create_table(self) -> list[list[float]]:

            n:int = len(self.X_points)
            table : List[List[float]] = [self.F_points.copy()]

            for i in range(n-1,0,-1):
                row : List[float] = []
                x_diss = n - i
                for j in range(i):
                    f = table[-1][j+1]
                    pf = table[-1][j]

                    xik = self.X_points[j+x_diss]
                    xi = self.X_points[j]

                    diff = round((f - pf)/(xik - xi), MAX_DIGITS)
                    row.append(diff)

                table.append(row)

            return  table

        def __update_table(self,x:float,f:float) -> list[list[float]]:
            n = len(self.F_points)

            table = self.diff_table.copy()
            table[0].append(f)

            for i in range(1, n - 1):

                f = table[i - 1][-1]
                pf = table[i - 1][-2]

                xi = self.X_points[-1 - i]

                diff = round((f - pf)/(x - xi), MAX_DIGITS)
                table[i].append(diff)

            f = table[- 1][-1]
            pf = table[- 1][-2]
            table.append([f - pf])

            return table

        def predict(self,x: float) -> float:
            super().predict(x)
            if self.is_forward:
                result = self.__predict_forward(x)
            else:
                result = self.__predict_backward(x)

            return result

        def __predict_forward(self,x:float) -> float:
            result = self.diff_table[0][0]
            newton_factor = 1
            n = len(self.diff_table)

            if DEBUG:
                print(f"p({x}) = {result} ", end='+')

            for k in range(1, n):

                factor = (x - self.X_points[k - 1])
                factor = round(factor, MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][0] * newton_factor, MAX_DIGITS)

                if DEBUG:
                    print(f" {self.diff_table[k][-1]} * ({newton_factor}) ", end='+')

            if DEBUG:
                print(f"\b= {result}")

            return result

        def __predict_backward(self, x: float) -> float:
            result = self.diff_table[0][-1]
            newton_factor = 1
            n = len(self.diff_table)

            if DEBUG:
                print(f"p({x}) = {result} ", end='+')

            for k in range(1, n):
                factor = (x - self.X_points[-k])
                factor = round(factor, MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][-1] * newton_factor, MAX_DIGITS)

                if DEBUG:
                    print(f" {self.diff_table[k][-1]} * ({newton_factor}) ", end='+')

            if DEBUG:
                print(f"\b= {result}")

            return result

        def add_point(self,x:float,f:float) -> None:
            super().add_point(x,f)

            self.MinX = min(x,self.MinX)
            self.MaxX = max(x,self.MaxX)

            self.X_points.append(x)
            self.F_points.append(f)

            self.diff_table = self.__update_table(x,f)

            if DEBUG:
                self.__print_table()

        def change_direction(self) -> None:
            self.is_forward = not self.is_forward
            if DEBUG:
                mode_str = "Forward"
                if not self.is_forward:
                    mode_str = "Backward"
                print(f"Changed to *({mode_str})* Newton Finite Differences Interpolation")

    def __init__(self,x_points:List[float],f_points:List[float],is_forward:bool = True):

        super().__init__(x_points=x_points,f_points=f_points,process_point=False)

        try:
            self.__interpolation = Newton.FiniteDifferences(x_points, f_points,is_forward)
        except DiffError:
            self.__interpolation = Newton.DividedDifferences(x_points,f_points,is_forward)
        except Exception as e:
            raise e

        if DEBUG:
            print("Newton init executed")

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

    def add_point(self,x:float,f:float) -> None:
        self.__interpolation.add_point(x,f)

class CubicSpline(BasicInterpolation):

    class SplinePolynomial:

        x: float = None
        a: float = None
        b: float = None
        c: float = None
        d: float = None

        def __init__(self,x:float = 0,a:float = 0,b:float = 0,c:float = 0,d:float = 0):
            self.x = x
            self.a = a
            self.b = b
            self.c = c
            self.d = d

        def __call__(self,x:float)-> float:
            return self.predict(x)

        def __str__(self) -> str:

            result  = f"{self.a}"
            result += f"+ {self.b} * (x - {self.x})"
            result += f"+ {self.c} * (x - {self.x})^2"
            result += f"+ {self.d} * (x - {self.x})^3"

            return result

        def __repr__(self) -> str:
            return f"CubicSpline.SplinePolynomial({self.x=},{self.a=},{self.b=},{self.c=},{self.d=})"

        def predict(self,x:float) -> float:
            factor = round(x - self.x,MAX_DIGITS)

            result = self.a
            result += round(self.b * (factor ** 1),MAX_DIGITS)
            result += round(self.c * (factor ** 2),MAX_DIGITS)
            result += round(self.d * (factor ** 3),MAX_DIGITS)

            return result

    __polynomials: List[SplinePolynomial] = []
    __n : int = 0

    def __init__(self,x_points:List[float], f_points:List[float],
                 dfx0:float = None,dfxn : float = None) -> None:

        super().__init__(x_points,f_points,process_point=False)

        self.__n = len(x_points)

        temp_x = [(x, f) for x, f in zip(x_points, f_points)]
        temp_x.sort(key=lambda point: point[0])


        self.X_points = []
        self.F_points = []

        for x, f in temp_x:
            self.X_points.append(x)
            self.F_points.append(f)


        self.MinX = self.X_points[0]
        self.MaxX = self.X_points[-1]

        self.__build_polynomials(dfx0,dfxn)

        if DEBUG:
            print("CubicSpline init executed")

    def __build_polynomials(self,dfx0:float = None,dfxn : float = None) -> None:

        h : List[float] = []
        self.__polynomials : List[CubicSpline.SplinePolynomial] = []

        for i in range(self.__n -1):

            x:float = self.X_points[i]
            a:float = self.F_points[i]
            polynomial = CubicSpline.SplinePolynomial(x=x,a=a)
            self.__polynomials.append(polynomial)

            hi :float = round(self.X_points[i+1] - self.X_points[i] ,MAX_DIGITS)
            h.append(hi)

        print(h)

        if dfx0 and dfxn:
            self.__polynomials[0 ].b = dfx0
            self.__polynomials[-1].b = dfxn
        else:
            self.__polynomials[0 ].c = 0
            self.__polynomials[-1].c = 0

        # Ac = y
        A : List[List[float]] = []
        y : List[float] = []

        for i in range(1,self.__n - 1):

            row :List[float] = [0 for _ in range(self.__n - 2)]
            index = i - 1

            if index > 0:
                row[ index-1 ] = h[index-1]

            row[ index ] = 2*(h[index-1] - h[index])

            if i != self.__n - 2:
                row[ index+1 ] = h[index]

            value : float = 0
            y.append(value)
            A.append(row)

        print('0000')
        for i in A:
            print(i)

    def predict(self,x: float) -> float:
        super().predict(x)

        for i in range(self.__n - 1):
            xi = self.X_points[i]
            xi1 = self.X_points[i + 1]

            if xi <= x < xi1:
                return self.__polynomials[i].predict(x)

        return self.F_points[-1]

    def add_point(self,x:float,f:float) -> None:
        raise Exception("not implemented")


def sc(n):
    x11 , f11 = [] , []
    for i in range(10,n+10):
        x11.append(i)
        f11.append(i**3)
    return x11,f11

if __name__ == "__main__":

    #print("Hello World")
    #x2 = Lagrange([1,2], [20,10])
    #print(x2(1.5))

    x12,f12 = sc(4)
    print(x12,f12[:3])
    cs = CubicSpline(x12,f12)
    print(cs(0))
    #x12 = [0,1,2]
    #f12 = [0,-1,2]

    #print(CC(1.5))
    #CC.add_point(10.5,10.5**3)

    # inp = input()
    # X , Y = [] , []
    # while inp != "":
    #    x , y = map(float,inp.split())
    #    X.append(x)
    #    Y.append(y)
    #    if len(X) > 1 :
    #        print(x - X[-2])
    #    inp = input()

    #x3 = x2.add_point(1.5,0)
    #print(x3(1.5))
