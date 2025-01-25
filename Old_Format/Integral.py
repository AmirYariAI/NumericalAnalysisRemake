import math

MAX_DIGITS = 10
Debug = True

def _inp_GetFunction():
    f_x = input('f(x) = ') 
    def f(x):
        return eval(f_x)
    
    a , b = map(float,input('(a,b) = ').strip().strip('(').strip(')').split(','))

    # function => f
    # range => (a,b)
    return f , (a,b)

def _inp_GetPoints():

    print("Enter your points : (xi,fi)")
    print("type (END) at the end of your points")
    
    x_points = []
    f_points = []

    point = input('( xi , fi ) : ')

    while point.upper() not in ("END",""):
        x,f = map(float,point.split())
        
        if x in x_points:
            raise Exception("Point is already in the list")
        if len(x_points) >= 3:
            #if (x - x_points[-1]) != (x_points[-1] - x_points[-2]):
            pass
            #    raise Exception("x is invalid")
            
        x_points.append(x)
        f_points.append(f)

        point = input('( xi , fi ) : ')
    
    if len(x_points) < 2:
        raise Exception("You Need at least two points")
    
    return x_points , f_points
   
def TrapezoidalRule(**kwargs):
    
    
    if "Debug" in kwargs.keys():
        Debug = kwargs["Debug"]
        kwargs.pop("Debug")
    else:
        Debug = False
    

    _sum = 0

    match kwargs:
        case {"function" : f , "range" : (a,b)} | {"f" : f , "range" : (a,b)} | {"f_x" : f , "range" : (a,b)}:
            
            
            if a >= b : raise ValueError(f"range ({a},{b}) is invalid ")
            
            h = 10 ** -MAX_DIGITS
            x = a
            n = math.ceil((b - a)/h)

            if Debug : print(f'{h=} {n=}')

            for _ in range(n):
                nx = x + h

                f_x = f(x)
                f_nx = f(nx)
                _sum += round((f_x + f_nx),MAX_DIGITS)

                x = nx
            
            _sum *= h/2
            _sum = round(_sum,MAX_DIGITS)

        case {"x_points":x_points,"f_points":f_points}:

            n = len(x_points)

            if len(f_points) != n: 
                raise ValueError(f"x_points and f_points must be at the same leangh")
            
            h = abs(x_points[1] - x_points[0])
            
            if Debug : print(f'{h=}')
    
            _sum = (2 *sum(f_points) - f_points[0] -  f_points[-1]) * (h/2)
            _sum = round(_sum,MAX_DIGITS)

        case _:
            raise ValueError("Input is invalid")
                
    return _sum

def SimpsonsRule(**kwargs):
    
    
    if "Debug" in kwargs.keys():
        Debug = kwargs["Debug"]
        kwargs.pop("Debug")
    else:
        Debug = False
    

    _sum = 0

    match kwargs:
        case {"function" : f , "range" : (a,b)} | {"f" : f , "range" : (a,b)} | {"f_x" : f , "range" : (a,b)}:
            
            h = 10 ** -MAX_DIGITS
            x = a 
            n = math.ceil((b - a)/(h*2))

            if Debug : print(f'{h=} {n=}')
            if a >= b or abs(b - a) <= h : raise ValueError(f"range ({a},{b}) is invalid ")
            
            for _ in range(n-1):
                nx = x + h
                n2x = x + 2*h

                f_x = f(x)
                f_nx = f(nx)
                f_n2x = f(n2x)

                area = (f_x +4*f_nx+ f_n2x) 
                area = round(area,MAX_DIGITS)
                _sum += area

                x = n2x
            
            _sum *= h/3
            _sum = round(_sum,MAX_DIGITS)

        case {"x_points":x_points,"f_points":f_points}:

            n = len(x_points) 

            if len(f_points) != n: raise ValueError(f"x_points and f_points must be at the same leangh")
            h = abs(x_points[1] - x_points[0])
            n = n //2
            if Debug : print(f'{h=}')
    
            for i in range(0,n):

                f_x = f_points[2*i + 0]
                f_nx = f_points[2*i + 1]
                f_n2x = f_points[2*i + 2]

                area = (f_x +4*f_nx+ f_n2x) 
                area = round(area,MAX_DIGITS)
                _sum += area


            _sum = _sum * h/3
            _sum = round(_sum,MAX_DIGITS)

        case _:
            raise ValueError("Input is invalid")
                
    return _sum

def MidpointMethod(**kwargs):
    
    
    if "Debug" in kwargs.keys():
        Debug = kwargs["Debug"]
        kwargs.pop("Debug")
    else:
        Debug = False
    

    _sum = 0

    match kwargs:
        case {"function" : f , "range" : (a,b)} | {"f" : f , "range" : (a,b)} | {"f_x" : f , "range" : (a,b)}:
            
            h = 10 ** -MAX_DIGITS
            x = a 
            n = math.ceil((b - a)/h)

            if Debug : print(f'{h=} {n=}')
            if a >= b or abs(b - a) <= h : raise ValueError(f"range ({a},{b}) is invalid ")
            
            for _ in range(n):
                nx = x + h / 2
                n2x = x + h

                f_nx = f(nx)

                area = f_nx 
                area = round(area,MAX_DIGITS)
                _sum += area

                x = n2x
            
            _sum *= 2*h
            _sum = round(_sum,MAX_DIGITS)

        case {"x_points":x_points,"f_points":f_points}:

            n = len(x_points) 

            if len(f_points) != n: raise ValueError(f"x_points and f_points must be at the same leangh")
            h = abs(x_points[1] - x_points[0])
            n = n //2
            if Debug : print(f'{h=}')
    
            for i in range(0,n):

                f_nx = f_points[2*i + 1]

                area = f_nx 
                area = round(area,MAX_DIGITS)
                _sum += area


            _sum *= 2*h
            _sum = round(_sum,MAX_DIGITS)

        case _:
            raise ValueError("Input is invalid")
                
    return _sum

def Simpsons38Rule(**kwargs):
    
    
    if "Debug" in kwargs.keys():
        Debug = kwargs["Debug"]
        kwargs.pop("Debug")
    else:
        Debug = False
    

    _sum = 0

    match kwargs:
        case {"function" : f , "range" : (a,b)} | {"f" : f , "range" : (a,b)} | {"f_x" : f , "range" : (a,b)}:
            
            h = 10 ** -MAX_DIGITS
            x = a 
            n = math.ceil((b - a)/(h*3))

            if Debug : print(f'{h=} {n=}')
            if a >= b or abs(b - a) <= h : raise ValueError(f"range ({a},{b}) is invalid ")
            
            _sum = round(f(x),MAX_DIGITS)

            for i in range(n):

                #f_x =  f(x + (3*i + 0)*h)
                f_nx = f(x + (3*i + 1)*h)
                f_n2x = f(x + (3*i + 2)*h)
                f_n3x = f(x + (3*i + 3)*h)

                area = 3*(f_nx+ f_n2x)
                if(i != 0):
                    area += 2*(f_n3x)
                area = round(area,MAX_DIGITS)
                _sum += area

                #x = n3x

            _sum += round(f(x + 3*n*h),MAX_DIGITS)
            _sum *= h*(0.375)
            _sum = round(_sum,MAX_DIGITS)

        case {"x_points":x_points,"f_points":f_points}:

            n = len(x_points) 

            if len(f_points) != n: raise ValueError(f"x_points and f_points must be at the same leangh")
            h = abs(x_points[1] - x_points[0])
            n = n //2
            if Debug : print(f'{h=}')
    
            for i in range(0,n):

                f_x = f_points[2*i + 0]
                f_nx = f_points[2*i + 1]
                f_n2x = f_points[2*i + 2]

                area = (f_x +4*f_nx+ f_n2x) 
                area = round(area,MAX_DIGITS)
                _sum += area


            _sum = _sum * h/3
            _sum = round(_sum,MAX_DIGITS)

        case _:
            raise ValueError("Input is invalid")
                
    return _sum
 
if __name__ == "__main__":

    print('Choose Your input type :')
    print('1.function')
    print('2.function table')
    inp = int(input('--> '))

    match inp:
        case 1:
            f , (a,b) = _inp_GetFunction()
            integral_input = dict(f_x = f , range = (a,b))
        case 2:
            x_points , f_points = _inp_GetPoints()
            integral_input = dict(x_points = x_points , f_points = f_points)
        case _ :
            raise ValueError(f'{inp=} is invalid')
        
    print('Choose Your method :')
    print("1.Trapezoidal Rule")
    print("2.Simpson's Rule")
    print("3.Midpoint Method")
    print("4.Simpsons 3/8 Rule")

    method = int(input('--> '))

    match method:
        case 1:
            result = TrapezoidalRule(**integral_input,Debug = Debug)
            print('-->',result)
        case 2:
            result = SimpsonsRule(**integral_input,Debug = Debug)
            print('-->',result)
        case 3:
            result = MidpointMethod(**integral_input,Debug = Debug)
            print('-->',result)
        case 4:
            result = Simpsons38Rule(**integral_input,Debug = Debug)
            print('-->',result)
        case _:
            raise ValueError(f"{method=} is invalid")