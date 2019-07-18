import multiprocessing
import time

def worker(item):
    time.sleep(1)
    return item

def handler():
    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(cpus)
    
    count = 0
    with open("teste.txt", 'r', encoding='utf-8') as t, open('out.txt', 'w', encoding='utf-8') as o:
        for r in p.imap(worker, t):
            count += 1
            print(r)
            o.write(r)

handler()