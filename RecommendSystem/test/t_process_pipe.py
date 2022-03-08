from array import array
from multiprocessing import Pool
import numpy

def return_dict():
    start = time.time()

    x = numpy.arange(0,1000000)
    end = time.time()

    print(x.nbytes/1024/1024/8)

    return x, end-start

import time

if __name__ == '__main__':
    # print(sys.getsizeof(a) / 1024 / 1024 / 8, 'MB')
    pool = Pool(4)
    dot = pool.apply_async(return_dict)
    start = time.time()
    a,b = dot.get()
    print(time.time() - start-b)
    # print(dot)
    dot = pool.apply_async(return_dict)
    start = time.time()
    a,b = dot.get()
    print(time.time() - start-b)


