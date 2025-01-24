import conf

def debug_on():
    conf.DEBUG = True

def debug_off():
    conf.DEBUG = False

def debug_status(status:str="auto"):

    match status.lower():
        case "auto":
            debug = conf.DEBUG
        case "true" | "on" | "1":
            debug = True
        case "false" | "off" | "0":
            debug = False
        case _:
            raise ValueError(f"{debug_status=} is invalid")

    return debug

def set_max_digits(x:int):
    conf.MAX_DIGITS = x

if __name__ == "__main__":
    print("DebugAssistant.py")