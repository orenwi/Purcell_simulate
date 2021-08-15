phidot = {"phidot1": 'sin',
        "phidot2": 'cos'}


def prs_fun(strfun,t):
match strfun:
    case 'sin':
        return math.sin(t)
    case 'cos':
        return math.cos(t)
    case 'sinShiftP':
        P = strfun[-1]
        return math.sin(t+P)
    case 'cosShiftP':
        P = strfun[-1]
        return math.cos(t+P)


    
print(phidot1)