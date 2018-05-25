import threading
import time


def get_goalfunctions(threadname, n):
    while True:
        print('hello from ({}), going to sleep for {} seconds'.format(threadname, n))
        time.sleep(n)
        print('woken up, your current goalfunction value is: {}'.format(goalfunction))



goalfunction = 0

t2 = threading.Thread(target=get_goalfunctions, name='thread2', args=('thread2', 5))
t2.daemon = True
t2.start()
# update goalfunction scope = main


def calculate():
    counter = 0
    while counter < 20:
        global goalfunction
        goalfunction += 5
        print('making calculations for 2 seconds')
        print(goalfunction)
        time.sleep(2)
        counter += 1

calculate()