import sys
import numpy as np
from multiprocessing import shared_memory
from multiprocessing import Pool
import time
from datetime import datetime


def add():
    r = 1 + 2
    return r


def numpy_array_sum(array):
    start = datetime.now()
    existing_shm = shared_memory.SharedMemory(name='psm_c11e264b')
    process_d = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
    print('process_d:', process_d)
    sum = process_d.sum()
    end = datetime.now()
    existing_shm.close()
    return sum, end - start


if __name__ == '__main__':
    shm_a = shared_memory.SharedMemory(create=True, size=10)
    print('shm_a:', sys.getsizeof(shm_a))
    buffer = shm_a.buf
    print('len shm_a:', len(buffer))

    a = np.array([1, 1, 2, 3, 5, 8])
    shm = shared_memory.SharedMemory(name='psm_c11e264b', create=True, size=a.nbytes)
    print('shm:', len(shm.buf))

    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    print('b:', b)
    print('shm name:', shm.name)

    existing_shm = shared_memory.SharedMemory(name='psm_c11e264b')
    c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
    print('c:', c)
    pool = Pool(4)
    sum_x = pool.apply_async(add)
    x = sum_x.get()
    print('sum_x: ', x)
    start = datetime.now()
    sum_d = pool.apply_async(numpy_array_sum, args=(b,))
    sum, cost = sum_d.get()
    end = datetime.now()
    print('sum_d:', sum, 'time:', cost, 'apply_async get() time cost:', end - start)



