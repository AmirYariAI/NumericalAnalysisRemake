import math
from typing import List
from DebugAssistant import debug_status
from EquationSolvers import EquationSolvers
from conf import *

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

    def __call__(self,x:float,debug_mode:str = "auto") -> float:
        return  self.predict(x,debug_mode)

    def predict(self,x: float,debug_mode:str = "auto") -> float:

        if not(self.MinX <= x <= self.MaxX):
            raise ValueError(f"{x=} must be in the range of [{self.MinX} , {self.MaxX}]")

        return 0.0

    def add_point(self,x:float,f:float) -> None:

        if x in self.X_points:
            raise  ValueError(f"{x=} is already in the list")

class DiffError(Exception):
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

    def predict(self,x : float,debug_mode:str = "auto") -> float:

        debug = debug_status(debug_mode)
        super().predict(x)

        fx : float =  0.0

        if debug:
            print(f"->> Lagrange Interpolation for p({x}):")

        for i,fi in enumerate(self.F_points):

            l = self.__lagrange_multiplier(x,i)
            fx += round(l * fi,MAX_DIGITS)

            if debug:
                print(f'({l})*{fi} ',end="+ ")

        fx = round(fx,MAX_DIGITS)

        if debug:
            print(f"\b\b= {fx}")

        return fx

    def add_point(self,x:float,f:float) -> None :

        super().add_point(x,f)

        self.X_points.append(x)
        self.F_points.append(f)

class Newton(BasicInterpolation):

    class FiniteDifferences(BasicInterpolation):

        diff_table : List[List[float]] = [[]]
        h : float = None
        is_forward:bool = True

        def __init__(self,x_points:List[float],f_points:List[float],is_forward:bool = True,debug_mode:str = "auto")-> None:

            debug = debug_status(debug_mode)
            
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
            if debug:
                self.__print_table()

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

        def predict(self,x: float,debug_mode:str = "auto") -> float:

            super().predict(x)
            if self.is_forward:
                result = self.__predict_forward(x,debug_mode)
            else:
                result = self.__predict_backward(x,debug_mode)

            return result

        def __predict_forward(self, x:float,debug_mode:str = "auto") -> float:

            debug = debug_status(debug_mode)

            result = self.diff_table[0][0]
            s = (x - self.MinX) / self.h
            s = round(s,MAX_DIGITS)
            newton_factor = 1
            n = len(self.diff_table)

            if debug:
                print(f"p({x}) = {result} ",end='+')

            for k in range(1,n):

                factor = (s-k+1) / k
                factor = round(factor,MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][0]*newton_factor,MAX_DIGITS)

                if debug:
                    print(f" {self.diff_table[k][0]} * ({newton_factor}) ", end='+')

            if debug:
                print(f"\b= {result}")

            return  result

        def __predict_backward(self, x:float,debug_mode:str = "auto") -> float:

            debug = debug_status(debug_mode)

            result = self.diff_table[0][-1]
            t = (x - self.MaxX) / self.h
            t = round(t, MAX_DIGITS)
            newton_factor = 1
            n = len(self.diff_table)

            if debug:
                print(f"p({x}) = {result} ",end='+')

            for k in range(1, n):
                factor = (t + k - 1) / k
                factor = round(factor, MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][-1] * newton_factor, MAX_DIGITS)

                if debug:
                    print(f" {self.diff_table[k][-1]} * ({newton_factor}) ", end='+')

            if debug:
                print(f"\b= {result}")

            return result

        def add_point(self,x:float,f:float,debug_mode:str = "auto") -> None:

            debug = debug_status(debug_mode)

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

            if debug:
                self.__print_table()

        def change_direction(self,debug_mode:str = "auto") -> None:

            debug = debug_status(debug_mode)

            self.is_forward = not self.is_forward
            if debug:
                mode_str = "Forward"
                if not self.is_forward:
                    mode_str = "Backward"
                print(f"Changed to *({mode_str})* Newton Finite Differences Interpolation")

    class DividedDifferences(BasicInterpolation):

        diff_table: List[List[float]] = [[]]
        is_forward: bool = True

        def __init__(self, x_points: List[float], f_points: List[float],is_forward: bool = True,debug_mode:str = "auto")-> None:

            debug = debug_status(debug_mode)

            super().__init__(x_points=x_points, f_points=f_points,process_point=True)
            self.is_forward = is_forward
            self.diff_table = self.__create_table()
            if debug:
                self.__print_table()

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

        def predict(self,x: float,debug_mode:str = "auto") -> float:
            super().predict(x)
            if self.is_forward:
                result = self.__predict_forward(x,debug_mode)
            else:
                result = self.__predict_backward(x,debug_mode)

            return result

        def __predict_forward(self,x:float,debug_mode:str = "auto") -> float:

            debug = debug_status(debug_mode)

            result = self.diff_table[0][0]
            newton_factor = 1
            n = len(self.diff_table)

            if debug:
                print(f"p({x}) = {result} ", end='+')

            for k in range(1, n):

                factor = (x - self.X_points[k - 1])
                factor = round(factor, MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][0] * newton_factor, MAX_DIGITS)

                if debug:
                    print(f" {self.diff_table[k][-1]} * ({newton_factor}) ", end='+')

            if debug:
                print(f"\b= {result}")

            return result

        def __predict_backward(self, x: float,debug_mode:str = "auto") -> float:

            debug = debug_status(debug_mode)

            result = self.diff_table[0][-1]
            newton_factor = 1
            n = len(self.diff_table)

            if debug:
                print(f"p({x}) = {result} ", end='+')

            for k in range(1, n):
                factor = (x - self.X_points[-k])
                factor = round(factor, MAX_DIGITS)

                newton_factor *= factor

                result += round(self.diff_table[k][-1] * newton_factor, MAX_DIGITS)

                if debug:
                    print(f" {self.diff_table[k][-1]} * ({newton_factor}) ", end='+')

            if debug:
                print(f"\b= {result}")

            return result

        def add_point(self,x:float,f:float,debug_mode:str = "auto") -> None:

            debug = debug_status(debug_mode)

            super().add_point(x,f)

            self.MinX = min(x,self.MinX)
            self.MaxX = max(x,self.MaxX)

            self.X_points.append(x)
            self.F_points.append(f)

            self.diff_table = self.__update_table(x,f)

            if debug:
                self.__print_table()

        def change_direction(self,debug_mode:str = "auto") -> None:

            debug = debug_status(debug_mode)

            self.is_forward = not self.is_forward
            if debug:
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

    def __repr__(self):
        return repr(self.__interpolation)
    def __str__(self):
        return  str(self.__interpolation)

    def __call__(self,x:float,debug_mode:str = "auto"):
        return self.__interpolation(x,debug_mode)

    def add_point(self,x:float,f:float,debug_mode:str = "auto") -> None:
        self.__interpolation.add_point(x,f,debug_mode)

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

        def __call__(self,x:float,debug_mode:str = "auto")-> float:
            return self.predict(x,debug_mode)

        def __str__(self) -> str:

            result  = f"S(x) = "
            result += f"({self.a}) "
            result += f"+ ({self.b}) * (x - {self.x}) "
            result += f"+ ({self.c}) * (x - {self.x})^2 "
            result += f"+ ({self.d}) * (x - {self.x})^3 "

            return result

        def __repr__(self) -> str:
            return f"CubicSpline.SplinePolynomial({self.x=},{self.a=},{self.b=},{self.c=},{self.d=})"

        def predict(self,x:float,debug_mode:str = "auto") -> float:

            debug = debug_status(debug_mode)

            factor = round(x - self.x,MAX_DIGITS)

            result = self.a
            result = round(result + round(self.b * (factor ** 1),MAX_DIGITS),MAX_DIGITS)
            result = round(result + self.c * (factor ** 2),MAX_DIGITS)
            result = round(result + self.d * (factor ** 3),MAX_DIGITS)



            if debug:
                print(f"S({x}) = ({self.a}) + ({self.b}) * ({x} - {self.x}) +",end='')
                print(f" ({self.c}) * ({x} - {self.x})^2 + ({self.d}) * ({x} - {self.x})^3 ",end='')
                print(f" = {result}")

            return result

    __polynomials: List[SplinePolynomial] = []
    __n : int = 0

    def __init__(self,x_points:List[float], f_points:List[float],debug_mode:str = "auto") -> None:

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

        self.__build_polynomials(debug_mode)

    def __build_polynomials(self,debug_mode:str = "auto") -> None:

        debug = debug_status(debug_mode)

        h : List[float] = []
        self.__polynomials : List[CubicSpline.SplinePolynomial] = []

        number_of_polynomial = self.__n - 1

        for i in range(number_of_polynomial):
            hi :float = round(self.X_points[i+1] - self.X_points[i] ,MAX_DIGITS)
            h.append(hi)

        c_factors: List[List[float]] = []
        y: List[float] = []

        for i in range(1,number_of_polynomial):

            row:List[float]= [0] * (number_of_polynomial - 1)

            index = i - 1
            if i != 1:
                row[index - 1] = h[i-1]
            row[index] = 2*(h[i - 1] + h[i])

            if i != number_of_polynomial - 1:
                row[index+1] = h[i]

            c_factors.append(row)

            a_i = self.F_points[i]
            a_ip1 = self.F_points[i + 1]
            a_im1 = self.F_points[i - 1]

            value = round(3/h[i] , MAX_DIGITS) * round(a_ip1 - a_i,MAX_DIGITS)
            value = round(value,MAX_DIGITS)

            value2 = round(3 / h[i - 1], MAX_DIGITS) * round(a_i - a_im1, MAX_DIGITS)
            value2 = round(value2, MAX_DIGITS)

            y.append(round(value - value2,MAX_DIGITS))

        c_values = [0] * (number_of_polynomial - 1)

        c_values = EquationSolvers.gauss_seidel(c_factors,c_values,y,debug_mode='off')
        c_values.insert(0,0)
        c_values.append(0)

        if debug:
            print("Spline Polynomials :")

        for i in range(number_of_polynomial):

            x: float = self.X_points[i]
            a: float = self.F_points[i]

            b_value1 : float = round(1/h[i],MAX_DIGITS) * round(self.F_points[i+1] - self.F_points[i],MAX_DIGITS)
            b_value1 = round(b_value1,MAX_DIGITS)
            b_value2 : float = round(h[i]/3,MAX_DIGITS) * round(2*c_values[i] - c_values[i + 1],MAX_DIGITS)
            b_value2 = round(b_value2,MAX_DIGITS)

            b:float = round(b_value1 - b_value2,MAX_DIGITS)

            c: float = c_values[i]

            d:float = round(round(c_values[i+1] - c_values[i],MAX_DIGITS) / (3*h[i]),MAX_DIGITS)

            polynomial = CubicSpline.SplinePolynomial(x=x, a=a , b=b,c=c,d=d)

            if debug :
                print(f"{i} => {str(polynomial).ljust(28 + (MAX_DIGITS + 3) * 6)}  , x in [{self.X_points[i]} , {self.X_points[i + 1]}]")
            self.__polynomials.append(polynomial)




    def predict(self,x: float,debug_mode:str = "auto") -> float:

        debug = debug_status(debug_mode)

        super().predict(x)

        for i in range(self.__n - 1):
            xi = self.X_points[i]
            xi1 = self.X_points[i + 1]

            if xi <= x < xi1:
                if debug:
                    print(f"{i} => ",end='')
                return self.__polynomials[i].predict(x,debug_mode)

        return self.F_points[-1]

    def add_point(self,x:float,f:float,debug_mode:str = "auto") -> None:
        raise Exception("not implemented")

class LinearRegression(BasicInterpolation):

    a_factors : List[float] = []
    k : int = None

    def __init__(self,x_points:List[float],f_points:List[float],k:int,debug_mode:str = "auto") -> None:

        super().__init__(x_points=x_points,f_points=f_points,process_point=True)
        self.k = k

        self.__build(debug_mode = debug_mode)

    def predict(self,x: float,debug_mode:str = "auto") -> float:

        debug = debug_status(debug_mode)

        f_x : float = 0

        if debug:
            print(f"f({x}) ", end="=")

        for i in range(self.k, -1, -1):

            if debug:
                if i == 0:
                    print(f" {self.a_factors[i]} ",end="= ")
                elif i == 1:
                    print(f" {self.a_factors[i]} * {x} ", end="+")
                else:
                    print(f" {self.a_factors[i]} * {x}^{i} ", end="+")

            s = round(self.a_factors[i] * round(x**i,MAX_DIGITS) ,MAX_DIGITS)
            f_x = round(f_x + s , MAX_DIGITS)

        if debug:
            print(f_x)

        return f_x

    def __build(self,debug_mode:str = "auto") -> None:

        debug = debug_status(debug_mode)

        n = len(self.X_points)

        x_power_values : List[float] = [1.0] * n

        x_power_sum_values : List[float] = []
        answer_matrix: List[float] = [0] * (self.k + 1)

        for i in range(2*self.k + 1):
            x_power_sum_values.append(sum(x_power_values))

            if i == 2*self.k:
                continue
            if i <= self.k:
                answer_i = [x_power_values[i] * self.F_points[i] for i in range(n)]
                answer_matrix[self.k - i] = sum(answer_i)

            x_power_values = [x_power_values[i] * self.X_points[i] for i in range(n)]

        factor_matrix : List[List[float]] = []

        for i in range(self.k , -1 , -1):

            row : List[float] = []

            for j in range(i + self.k , i - 1 , -1):
                row.append(x_power_sum_values[j])

            factor_matrix.append(row)


        a = [0] * (self.k + 1)

        self.a_factors = EquationSolvers.gauss_seidel(factor_matrix,a,answer_matrix,debug_mode='off')
        self.a_factors.reverse()

        if debug:
            print("f(x) ",end="=")
            for i in range(self.k,-1,-1):
                if i == 0:
                    print(f" {self.a_factors[i]}  ,MSE : {self.mean_squared_error()}")
                    break
                elif i == 1:
                    print(f" {self.a_factors[i]} * x ", end="+")
                    continue

                print(f" {self.a_factors[i]} * x^{i} ",end="+")

    def mean_squared_error(self) -> float:

        mse :float = 0
        n : int = len(self.X_points)

        for x,f in zip(self.X_points,self.F_points):

            f_x = self.predict(x,debug_mode="off")

            error = round(f - f_x , MAX_DIGITS)
            squared_error = round(error ** 2 , MAX_DIGITS)

            mse = round(mse + squared_error , MAX_DIGITS)

        mse = round(mse / n , MAX_DIGITS)

        return mse

if __name__ == "__main__":

    test_x,test_f = [0 , 0.05 , 0.2] , [1 , 2, 3]

    cs = CubicSpline(test_x,test_f)
    print(cs(1))
