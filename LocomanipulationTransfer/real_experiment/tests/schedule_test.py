# !usr/bin/env python
# -*- coding:utf-8 _*-
import time
import torch

time_start = time.time_ns()

# SuperFastPython.com
# example of using a thread timer object
from threading import Timer, Thread
 
#t1 = None
# target task function
def task(message):
    time_elasped_ms = (time.time_ns() - time_start)/1e+6
    time_elasped_ms = round(time_elasped_ms)
    print("Time in milliseconds: ", time_elasped_ms)
    time.sleep(0.01)
    time_elasped_ms = (time.time_ns() - time_start)/1e+6
    time_elasped_ms = round(time_elasped_ms)
    print("Time in milliseconds: ", time_elasped_ms)
    print("****")

def timer_task():
    t1 = Thread(target=task, args=(10,))
    t1.start()
    run_timer()
 
def run_timer():
    # create a thread timer object
    timer = Timer(0.0330, timer_task)
    # start the timer object
    timer.start()

#t1 = Thread(target=task, args=(10,))
run_timer()
