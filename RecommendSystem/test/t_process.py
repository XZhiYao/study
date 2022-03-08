# -*- coding:utf-8 -*-
import os, time
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing import shared_memory
from datetime import datetime

# def divBlock_matrix(matrix):



def divBlock_sequence(block_num, thread_num):
    # block_num = [1, 2, ... 15, 16]
    block = []
    for i in range(0, thread_num):  # 0~3
        for j in range(thread_num):  # 0~3
            print('j:', i * thread_num + j)
            block.append(block_num[j * thread_num + (i + j) % thread_num])
    print('block: ', block)


def long_time_task(name, num):
    print('Run task %s (%s)...' % (name, os.getpid()))

    start = time.time()
    num += 1
    time.sleep(1)

    end = time.time()

    print('Task %s runs %0.2f seconds.' % (name, (end - start)))
    return num


class MyProcess1(Process):
    def __init__(self):
        Process.__init__(self)

    def run(self):
        print("子进程1开始>>> pid={0},ppid={1}".format(os.getpid(), os.getppid()))
        start1 = datetime.now()
        time.sleep(8)
        end1 = datetime.now()
        print("Process1 time cost:", end1 - start1)
        print("子进程1终止>>> pid={}".format(os.getpid()))


class MyProcess2(Process):
    def __init__(self):
        Process.__init__(self)

    def run(self):
        print("子进程2开始>>> pid={0},ppid={1}".format(os.getpid(), os.getppid()))
        start2 = datetime.now()
        time.sleep(5)
        end2 = datetime.now()
        print("Process2 time cost:", end2 - start2)
        print("子进程2终止>>> pid={}".format(os.getpid()))


def main():
    print("主进程1开始>>> pid={}".format(os.getpid()))
    start = datetime.now()
    myp1 = MyProcess1()
    myp1.start()
    myp2 = MyProcess2()
    myp2.start()
    myp1.join()
    end = datetime.now()
    print("主进程1终止, time:", end - start)

    print("主进程2开始>>> pid={}".format(os.getpid()))
    p = Pool(5)
    num = []
    a = 0
    for i in range(10):
        n = p.apply_async(long_time_task, args=(i, a))
        n.get()
        # num.append(n)
    # for  n in num:
    #     print(n.get())

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('total num list:', num)
    print('All subprocesses done.')

    print('Parent process %s.' % os.getpid())


if __name__ == '__main__':
    # main()
    block_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    divBlock_sequence(block_num, 4)
