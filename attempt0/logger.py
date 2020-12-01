import datetime


# This turns all information to a string
def arg_to_string(*args, **kwargs):
    s = ""
    for a in args:
        if type(a) == tuple or type(a) == list:
            s += arg_to_string(*a)
        else:
            s += str(a)
    return s


# This prints and writes down all information
def loggyboi(*args, **kwargs):
    data = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)") + arg_to_string(args)
    print(data)
    f = open("log.txt", "a")
    f.write(data + "\n")
    f.close()