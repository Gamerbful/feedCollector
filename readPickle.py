import pickle
res = None
with (open("mypicklefile", "rb")) as openfile:
    while True:
        try:
            res = pickle.load(openfile)
        except EOFError:
            break


print(res[0]['06cfebaa3805686ccf7e91ec8b1d0d33a2b8b310f99855a4e1a1127475155b61']['data'])
