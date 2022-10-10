import pickle
res = None
with (open("mypicklefile", "rb")) as openfile:
    while True:
        try:
            res = pickle.load(openfile)
        except EOFError:
            break


print(len(res))