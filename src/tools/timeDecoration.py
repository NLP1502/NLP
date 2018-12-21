# Copyright 2018-YuejiaXiang, NLP Lab., Northeastern university
# This decoration can calculate time-cost of function.
#
# How to use:
# from timeDecoration import clock
# @clock
# def function(): ...
#
import time, timeit
def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        #print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        print('[%0.8fs] %s' % (elapsed, name))
        return result
    return clocked