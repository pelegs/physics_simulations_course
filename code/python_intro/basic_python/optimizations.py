# ------- Loops ------- #
from timeit import timeit, Timer
from numpy import arange, sum as npsum


def sum_while(n):
    i = wsum = 0
    while i < n:
        wsum += i
        i += 1
    return wsum


def sum_for(n):
    fsum = 0
    for i in range(n):
        fsum += i
    return fsum


def sum_direct(n):
    return sum(range(n))


def sum_numpy(n):
    return npsum(range(n))


def sum_np_arange(n):
    return npsum(arange(0, n))


N = 100000
t1 = Timer(lambda: sum_while(N))
t2 = Timer(lambda: sum_for(N))
t3 = Timer(lambda: sum_direct(N))
t4 = Timer(lambda: sum_numpy(N))
t5 = Timer(lambda: sum_np_arange(N))

# report
reps = 50
print(f"""
    sum_while:     {t1.timeit(reps)},
    sum_for:       {t2.timeit(reps)},
    sum_direct:    {t3.timeit(reps)},
    sum_numpy:     {t4.timeit(reps)},
    sum_np_arange: {t5.timeit(reps)}.
""")


# ------- List comp. vs. append ------- #
from timeit import timeit, Timer

def append(n):
    lst = []
    for n in range(n):
        lst.append(n)
    return lst


def comprh(n):
    return [n for n in range(n)]


N = 100000
t1 = Timer(lambda: append(N))
t2 = Timer(lambda: comprh(N))

# report
reps = 200
print(f"""
    append:        {t1.timeit(reps)},
    comprhension:  {t2.timeit(reps)}.
""")


# ------- Generators ------- #
def generator():
    yield 1
    yield 2
    yield 3
    yield 4
    return "test"

print(generator())
for x in generator():
    print(x)

# x = generator()
# print(x[0])


from timeit import timeit, Timer
from numpy.random import uniform


def sqr_list(nums):
    sqrs = []
    for n in nums:
        sqrs.append(n**2)
    return sqrs


def sqr_gen(nums):
    for n in nums:
        yield n**2


# Measure run time
numbers = uniform(size=10000)
# print(numbers[1337])
t1 = Timer(lambda: sqr_list(numbers))
t2 = Timer(lambda: sqr_gen(numbers))
t3 = Timer(lambda: [x for x in sqr_gen(numbers)])
reps = 100  # List should be faster
# reps = 1000 # Generator should be faster
iterator = t1.timeit(reps)
generator = t2.timeit(reps)
print(f"""
iterator:  {t1.timeit(reps)},
generator: {t2.timeit(reps)}.
...actual generator: {t3.timeit(reps)}.
""")


from pathlib import Path
p = Path('.').rglob('*.txt')
print(f'p is of type {type(p)}')
print('--------------------------')
for file in p:
    print(file)


# Generator comprhension
