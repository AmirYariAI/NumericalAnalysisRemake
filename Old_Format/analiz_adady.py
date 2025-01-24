import math

MAX_DIGITS = 10
DEBUG_MODE = True


# Dubug Tabel Functions

def CreateDebugTable(f_name: str):
    tabel_space = 4 + 4 * (MAX_DIGITS + 4) + 15 + 7 + 10
    print(f'(( {f_name} ))'.center(tabel_space))

    print(f"n".center(4), end="|")
    print(f"a".center(MAX_DIGITS + 4), end="|")
    print(f"b".center(MAX_DIGITS + 4), end="|")
    print(f"x".center(MAX_DIGITS + 4), end="|")
    print(f"f(x)".center(MAX_DIGITS + 4), end="|")
    print(f"f(a)f(x) sign".center(15), end="|")
    print(f"stop".center(7), end="|")
    print(f"Note".center(10))


def AddDebugTableRow(n, a, b, x, f_x, sign, stop, note=""):
    print(f"{n}".center(4), end="|")
    print(f"{a}".center(MAX_DIGITS + 4), end="|")
    print(f"{b}".center(MAX_DIGITS + 4), end="|")
    print(f"{x}".center(MAX_DIGITS + 4), end="|")
    print(f"{f_x}".center(MAX_DIGITS + 4), end="|")
    print(f"{'+' if sign else '-'}".center(15), end="|")
    print(f"{stop}".center(7), end="|")
    print(f"{note}".center(10))


# Alghoritms

def Bisectionmethod(f, range, stop_function, Debug=True):
    a = range[0]
    b = range[1]

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    px = 0
    x = (a + b) / 2
    x = round(x, MAX_DIGITS)
    f_x = round(f(x), MAX_DIGITS)

    n = 1

    if Debug:
        CreateDebugTable(Bisectionmethod.__name__)
        AddDebugTableRow(n, a, b, x, f_x, (f(a) * f(x) < 0), False)

    while not stop_function(f, n, px, x):

        f_a = round(f(a), MAX_DIGITS)

        f_value = round(f_a * f_x, MAX_DIGITS)

        if (f_value == 0):
            return x
        elif (f_value < 0):
            b = x
        else:
            a = x

        px = x
        x = (a + b) / 2
        x = round(x, MAX_DIGITS)

        f_x = round(f(x), MAX_DIGITS)
        n = n + 1

        if Debug:
            AddDebugTableRow(n, a, b, x, f_x, (f_value < 0), stop_function(f, n, px, x))

    return x


def RegulaFalsi(f, range, stop_function, Debug=False):
    a = range[0]
    b = range[1]

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    f_a = round(f(a), MAX_DIGITS)
    f_b = round(f(b), MAX_DIGITS)

    px = 0
    x = (a * f_b - b * f_a) / (f_b - f_a)
    x = round(x, MAX_DIGITS)
    f_x = round(f(x), MAX_DIGITS)
    n = 1

    if Debug:
        CreateDebugTable(RegulaFalsi.__name__)
        AddDebugTableRow(n, a, b, x, f_x, (f(a) * f(x) < 0), False)

    while not stop_function(f, n, px, x):

        f_value = round(f_a * f_x, MAX_DIGITS)
        if (f_value == 0):
            return x
        if (f_value < 0):
            b = x
        else:
            a = x

        f_a = round(f(a), MAX_DIGITS)
        f_b = round(f(b), MAX_DIGITS)

        px = x
        x = (a * f_b - b * f_a) / (f_b - f_a)
        x = round(x, MAX_DIGITS)
        f_x = round(f(x), MAX_DIGITS)

        n = n + 1

        if Debug:
            AddDebugTableRow(n, a, b, x, f_x, (f_value < 0), stop_function(f, n, px, x))

    return x


def OptimizedRegulaFalsi(f, range, stop_function, Debug=False):
    a = range[0]
    b = range[1]

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    f_a = round(f(a), MAX_DIGITS)
    f_b = round(f(b), MAX_DIGITS)

    px = 0
    x = (a * f_b - b * f_a) / (f_b - f_a)
    x = round(x, MAX_DIGITS)
    f_x = round(f(x), MAX_DIGITS)
    n = 1
    f_value = round(f_a * f_x, MAX_DIGITS)

    if Debug:
        CreateDebugTable(OptimizedRegulaFalsi.__name__)
        AddDebugTableRow(n, a, b, x, f_x, (f(a) * f(x) < 0), False)

    value_count = {'a': 0, 'b': 0}
    note = ""

    while not stop_function(f, n, px, x):

        if (f_value < 0):
            value_count['a'] += 1
            value_count['b'] = 0
            b = x
            note = f"n(a) = {value_count['a']}"
        else:
            value_count['b'] += 1
            value_count['a'] = 0
            a = x
            note = f"n(b) = {value_count['b']}"

        f_a = round(f(a), MAX_DIGITS)
        f_b = round(f(b), MAX_DIGITS)

        if (value_count['b'] >= 2):
            f_b = f_b / 2
            value_count = {'a': 0, 'b': 0}
            note = "f(b) / 2"
        elif (value_count['a'] >= 2):
            f_a = f_a / 2
            value_count = {'a': 0, 'b': 0}
            note = "f(a) / 2"

        px = x
        x = (a * f_b - b * f_a) / (f_b - f_a)
        x = round(x, MAX_DIGITS)
        f_x = round(f(x), MAX_DIGITS)

        f_value = round(f_a * f_x, MAX_DIGITS)
        n = n + 1

        if (f_value == 0):
            if Debug:
                AddDebugTableRow(n, a, b, x, f_x, (f_value < 0), True, "f(a)f(x)=0")
            return x

        if Debug:
            AddDebugTableRow(n, a, b, x, f_x, (f_value < 0), stop_function(f, n, px, x), note)

    return x


def FixedPointmethod(g, px, range, stop_function, Check_g=False, Debug=False):
    a = range[0]
    b = range[1]

    if Check_g:
        if not InRange(g, range):
            raise Exception(f"g(x) is not in [{a},{b}]")

        if not DerivativeInRange(g, range):
            raise Exception(f"|g'(x)| is more than (or equal to) 1")

    def f(x):
        return g(x) - x

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    x = round(g(px), MAX_DIGITS)
    x = round(x, MAX_DIGITS)

    n = 1

    if Debug:
        CreateDebugTable(FixedPointmethod.__name__)

        f_x = round(f(x), MAX_DIGITS)
        AddDebugTableRow(n, a, b, x, f_x, False, False)

    while not stop_function(f, n, px, x):

        px = x
        x = round(g(px), MAX_DIGITS)
        x = round(x, MAX_DIGITS)

        n = n + 1

        if Debug:
            f_x = round(f(x), MAX_DIGITS)
            AddDebugTableRow(n, a, b, x, f_x, False, stop_function(f, n, px, x))

    return x


def OptimizedFixedPointmethod(g, px, range, stop_function, Check_g=False, Debug=False):
    a = range[0]
    b = range[1]

    if Check_g:
        if not InRange(g, range):
            raise Exception(f"g(x) is not in [{a},{b}]")

        if not DerivativeInRange(g, range):
            raise Exception(f"|g'(x)| is more than (or equal to) 1")

    x_landa = (a + b) / 2
    g_landa_derivative = (g(x_landa + 10 ** -MAX_DIGITS) - g(x_landa)) / 10 ** -MAX_DIGITS
    landa = round(g_landa_derivative, MAX_DIGITS)

    def new_g(x):
        y = (g(x) + (landa * x)) / (1 + landa)
        return y

    def f(x):
        return g(x) - x

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    x = round(new_g(px), MAX_DIGITS)
    x = round(x, MAX_DIGITS)

    n = 1

    if Debug:
        CreateDebugTable(OptimizedFixedPointmethod.__name__)

        f_x = round(f(x), MAX_DIGITS)
        AddDebugTableRow(n, a, b, x, f_x, False, False, landa)

    while not stop_function(f, n, px, x):

        px = x
        x = round(new_g(px), MAX_DIGITS)
        x = round(x, MAX_DIGITS)

        n = n + 1

        if Debug:
            f_x = round(f(x), MAX_DIGITS)
            AddDebugTableRow(n, a, b, x, f_x, False, stop_function(f, n, px, x))

    return x


def Newtonmethod(f, px, range, stop_function, Debug=False):
    a = range[0]
    b = range[1]

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    d_f = round((f(px + 10 ** -MAX_DIGITS) - f(px)) / 10 ** -MAX_DIGITS, MAX_DIGITS)
    f_m_f_d = round(f(px) / d_f, MAX_DIGITS)
    x = round(px - f_m_f_d, MAX_DIGITS)
    n = 1

    if Debug:
        CreateDebugTable(Newtonmethod.__name__)

        f_x = round(f(x), MAX_DIGITS)
        AddDebugTableRow(n, a, b, x, f_x, False, False)

    while not stop_function(f, n, px, x):

        px = x
        d_f = round((f(px + 10 ** -MAX_DIGITS) - f(px)) / 10 ** -MAX_DIGITS, MAX_DIGITS)
        f_m_f_d = round(f(px) / d_f, MAX_DIGITS)
        x = round(px - f_m_f_d, MAX_DIGITS)

        n = n + 1

        if Debug:
            f_x = round(f(x), MAX_DIGITS)
            AddDebugTableRow(n, a, b, x, f_x, False, stop_function(f, n, px, x))

    return x


def Secantmethod(f, range, stop_function, Debug=False):
    a = range[0]
    b = range[1]

    if not (f(a) * f(b) < 0):
        raise Exception(f"{range=} is Invalid")

    f_a = round(f(a), MAX_DIGITS)
    f_b = round(f(b), MAX_DIGITS)

    px = b
    x = (a * f_b - b * f_a) / (f_b - f_a)
    x = round(x, MAX_DIGITS)
    f_x = round(f(x), MAX_DIGITS)
    n = 1

    if Debug:
        CreateDebugTable(Secantmethod.__name__)
        AddDebugTableRow(n, a, b, x, f_x, (f(a) * f(x) < 0), False)

    while not stop_function(f, n, px, x):

        f_value = round(f_a * f_x, MAX_DIGITS)

        a = b
        b = x

        f_a = round(f(a), MAX_DIGITS)
        f_b = round(f(b), MAX_DIGITS)

        px = x
        x = (a * f_b - b * f_a) / (f_b - f_a)
        x = round(x, MAX_DIGITS)
        f_x = round(f(x), MAX_DIGITS)

        n = n + 1

        if Debug:
            AddDebugTableRow(n, a, b, x, f_x, (f_value < 0), stop_function(f, n, px, x))

    return x


# Others

def InRange(g, range, epsilon=10 ** (-MAX_DIGITS)):
    a = range[0]

    while a <= b:

        g_x = round(g(a), MAX_DIGITS)

        if not (a <= g_x <= b):
            return False

        a = a + epsilon

    return True


def DerivativeInRange(g, range, epsilon=10 ** (-MAX_DIGITS)):
    a = range[0]

    while a <= b:

        g_x = round(g(a), MAX_DIGITS)
        g_xe = round(g(a + epsilon), MAX_DIGITS)

        Derivative = (g_xe - g_x) / (epsilon)
        Derivative = round(Derivative, MAX_DIGITS)

        if not (abs(Derivative) < 1):
            return False

        a = a + epsilon

    return True


def StopFunction(f, n, px, x):
    # Function Setting
    stop_method = 4
    epsilon = 1e-5
    max_n = 150
    r = 0

    match stop_method:
        case 0:
            return abs(round(x - r, MAX_DIGITS)) < epsilon
        case 1:
            return abs(round(f(x), MAX_DIGITS)) < epsilon
        case 2:
            return abs(round(x - px, MAX_DIGITS)) < epsilon
        case 3:
            return n >= max_n
        case 4:  # 3 or 1
            return (n >= max_n) or (abs(round(f(x), MAX_DIGITS)) < epsilon)
        case 5:  # 3 or 2
            return (n >= max_n) or (abs(round(x - px, MAX_DIGITS)) < epsilon)
        case _:
            raise Exception(f"{stop_method=} value is not valid")


def build_f(eq: str):
    def f(x):
        return eval(eq.lower())

    return f


def build_StopFunction(stop_method=4, epsilon=10 ** (-MAX_DIGITS), max_n=50, r=0):
    def StopFunction(f, n, px, x):

        match stop_method:
            case 0:
                return abs(round(x - r, MAX_DIGITS)) < epsilon
            case 1:
                return abs(round(f(x), MAX_DIGITS)) < epsilon
            case 2:
                return abs(round(x - px, MAX_DIGITS)) < epsilon
            case 3:
                return n >= max_n
            case 4:  # 3 or 1
                return (n >= max_n) or (abs(round(f(x), MAX_DIGITS)) < epsilon)
            case 5:  # 3 or 2
                return (n >= max_n) or (abs(round(x - px, MAX_DIGITS)) < epsilon)
            case _:
                raise Exception(f"{stop_method=} value is not valid")

    return StopFunction


if __name__ == "__main__":

    eq = input('f(x) = ')
    [a, b] = map(float, input('[a,b] = ').split())
    f = build_f(eq)

    function_inputs = {}

    print('Stop Condition :')
    print('0.|x - r| < epsilon')
    print('1.|f(a)|  < epsilon')
    print('2.|x_n - x_n-1|  < epsilon')
    print('3.n >= max_n')
    print('4. 3 or 1')
    print('5. 3 or 2')

    default_stop_method = 1
    stop_method = input(f'Choose Your Stop Method [{default_stop_method}]--> ')
    stop_method = int(stop_method) if (stop_method != '') else default_stop_method
    function_inputs['stop_method'] = int(stop_method)

    epsilon = input('Epsilon (Enter = Skip)--> ')
    if (epsilon != ''):
        function_inputs['epsilon'] = float(epsilon)

    max_n = input('Maximom n (Enter = Skip)--> ')
    if (max_n != ''):
        function_inputs['max_n'] = int(max_n)

    r = input('r (Enter = Skip)--> ')
    if (r != ''):
        function_inputs['r'] = float(r)

    StopFunction = build_StopFunction(**function_inputs)
    StopFunction_2 = build_StopFunction(stop_method=3, max_n=2)

    print('((Alghoritms))'.center(30))
    print(f'1.{Bisectionmethod.__name__}')
    print(f'2.{RegulaFalsi.__name__}')
    print(f'3.{OptimizedRegulaFalsi.__name__}')
    print(f'4.{FixedPointmethod.__name__}')
    print(f'5.{OptimizedFixedPointmethod.__name__}')
    print(f'6.{Newtonmethod.__name__}')
    print(f'7.{Secantmethod.__name__}')

    default_algorithm = 1
    algorithm = input(f'--> Choose Your Alghoritm [{default_algorithm}]: ')
    algorithm = int(algorithm) if (algorithm != '') else default_algorithm

    match algorithm:
        case 1:
            x = Bisectionmethod(f, [a, b], StopFunction, DEBUG_MODE)
        case 2:
            x = RegulaFalsi(f, [a, b], StopFunction, DEBUG_MODE)
        case 3:
            x = OptimizedRegulaFalsi(f, [a, b], StopFunction, DEBUG_MODE)
        case 4:

            print('((Alghoritms))'.center(30))
            print(f'1.{Bisectionmethod.__name__}')
            print(f'2.{RegulaFalsi.__name__}')
            print(f'3.{OptimizedRegulaFalsi.__name__}')

            default_x0_algorithm = 1
            x0_algorithm = input(f'--> Choose Your x_0 Alghoritm [{default_x0_algorithm}]:')
            x0_algorithm = int(x0_algorithm) if (x0_algorithm != '') else default_x0_algorithm


            def f2(x):
                return f(x) - x


            match x0_algorithm:
                case 1:
                    x0 = Bisectionmethod(f2, [a, b], StopFunction_2, DEBUG_MODE)
                case 2:
                    x0 = RegulaFalsi(f2, [a, b], StopFunction_2, DEBUG_MODE)
                case 3:
                    x0 = OptimizedRegulaFalsi(f2, [a, b], StopFunction_2, DEBUG_MODE)

            x = FixedPointmethod(f, x0, [a, b], StopFunction, Debug=DEBUG_MODE)
        case 5:

            print('((Alghoritms))'.center(30))
            print(f'1.{Bisectionmethod.__name__}')
            print(f'2.{RegulaFalsi.__name__}')
            print(f'3.{OptimizedRegulaFalsi.__name__}')

            default_x0_algorithm = 1
            x0_algorithm = input(f'--> Choose Your x_0 Alghoritm [{default_x0_algorithm}]:')
            x0_algorithm = int(x0_algorithm) if (x0_algorithm != '') else default_x0_algorithm


            def f2(x):
                return f(x) - x


            match x0_algorithm:
                case 1:
                    x0 = Bisectionmethod(f2, [a, b], StopFunction_2, DEBUG_MODE)
                case 2:
                    x0 = RegulaFalsi(f2, [a, b], StopFunction_2, DEBUG_MODE)
                case 3:
                    x0 = OptimizedRegulaFalsi(f2, [a, b], StopFunction_2, DEBUG_MODE)

            x = OptimizedFixedPointmethod(f, x0, [a, b], StopFunction, Debug=DEBUG_MODE)
        case 6:

            print('((Alghoritms))'.center(30))
            print(f'1.{Bisectionmethod.__name__}')
            print(f'2.{RegulaFalsi.__name__}')
            print(f'3.{OptimizedRegulaFalsi.__name__}')

            default_x0_algorithm = 1
            x0_algorithm = input(f'--> Choose Your x_0 Alghoritm [{default_x0_algorithm}]:')
            x0_algorithm = int(x0_algorithm) if (x0_algorithm != '') else default_x0_algorithm

            match x0_algorithm:
                case 1:
                    x0 = Bisectionmethod(f, [a, b], StopFunction_2, DEBUG_MODE)
                case 2:
                    x0 = RegulaFalsi(f, [a, b], StopFunction_2, DEBUG_MODE)
                case 3:
                    x0 = OptimizedRegulaFalsi(f, [a, b], StopFunction_2, DEBUG_MODE)

            x = Newtonmethod(f, x0, [a, b], StopFunction, DEBUG_MODE)
        case 7:
            x = Secantmethod(f, [a, b], StopFunction, DEBUG_MODE)
        case _:
            raise Exception("Invalid Input")

    print(f'x = {x} ({MAX_DIGITS} S)')