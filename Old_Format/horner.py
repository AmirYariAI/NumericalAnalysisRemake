import time

def Normal_method(f,x):
    
    normal_start = time.time()
    
    p_x = 0
    for i in range(len(f)):
        p_x += f[i] * (x ** i)
    time.sleep(0.5)
    return p_x

def Horner_method(x):

    p_x = 0
    for i in range(len(f)-1,-1,-1):
        p_x = p_x *x +f[i]
    time.sleep(0.5)
     
    return p_x

if __name__ == "__main__":

    print('index : i => a * x**i')
    f = list(map(float,input('--> ').split()))
    x = float(input('--> '))

    #f = [10,5,3,23,12,123.78]

    normal_start = time.time()
    p_x_n = Normal_method(f,x)
    normal_end = time.time()

    horner_start = time.time()
    p_x_h = Horner_method(x)
    horner_end = time.time()

    print(f'Normal : p(x) = {p_x_n} ({normal_end - normal_start - 0.5})')
    print(f'Horner : p(x) = {p_x_h} ({horner_end - horner_start - 0.5})')