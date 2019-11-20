#log = extern_func("console", "log", (int,))


#def logplus1(a: int):
#    log(a + 1)


def add(a: int, b: int) -> int:
    c = a - b
#    log(c)
#    logplus1(c)
    return c * 2
