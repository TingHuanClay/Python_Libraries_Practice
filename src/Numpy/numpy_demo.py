import time
import numpy as np


def test_numpy_speed():
    x = np.random.random(100000000)

    """
    Count Mean by calculation
    """
    start = time.time()
    sum(x)/len(x)
    print(time.time() - start)

    """
    Count Mean by Numpy is more efficient
    """
    start = time.time()
    np.mean(x)
    print(time.time() - start)


def np_array_demo():
    x = np.array([1, 2, 3, 4, 5])
    print(x)
    print(type(x))
    print(x.dtype)
    print(x.shape)

    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(y)
    print(y.shape)
    print(y.size)

    x = np.array(['Hello', 'World'])
    print(x)
    print(f"shape: {x.shape}")
    print(f"type: {type(x)}")
    print(f"dtype: {x.dtype}")

    x = np.array([1, 2, 'World'])
    print(x)
    print(f"shape: {x.shape}")
    print(f"type: {type(x)}")
    print(f"dtype: {x.dtype}")

    """
    Assign the np.array to specific dtype
    which is helpful to casting the data automatically
    """
    x = np.array([1.5, 2.2, 3.6], dtype=np.int64)
    print()
    print(x)
    print(f"dtype:{x.dtype}")


def numpy_default_values_demo():
    my_list = [1, 2, 3, 4, 5]
    arr = np.array(my_list)
    print(arr)

    """
    Initialize the np array with 0
    """
    x = np.zeros((3, 4))
    print(x)
    print(f"dtype:{x.dtype}")

    """
    Initialize the np array with 0 and specific dtype
    """
    x = np.zeros((3, 4), dtype=int)
    print(x)
    print(f"dtype:{x.dtype}")

    x = np.ones((3, 4))
    print(x)

    x = np.full((4, 3), 6)
    print(x)
    print(f"dtype:{x.dtype}")

    x = np.full((4, 3), 7.0)
    print(x)
    print(f"dtype:{x.dtype}")

    """
    TWO ways to Initialize with diagonal vlues
    """
    x = np.eye(5)
    print(x)

    x = np.diag([10.0, 20.0, 30.0, 50.0])
    print(x)

    """
    3 ways to Initialize the array with specific number period
    """
    print()
    # 1. from 0 to the n-1
    x = np.arange(10)
    print(x)
    # 2. from begining to the end-1
    x = np.arange(4, 10)
    print(x)
    # 3. from begining to the end-1 and the difference between each is assigned
    x = np.arange(0, 20, 4)
    print(x)

    """
    """
    print()
    # create n value between start and the end
    # (create 10 value between 0 and the 25)
    x = np.linspace(0, 25, 10)
    print(x)
    # without endpoint
    x = np.linspace(0, 25, 10, endpoint=False)
    print(x)

    """
    Reorganize the np array to the dimensions specified, but the total count has to be the same
    """
    print()
    x = np.arange(0, 20)
    print(x)
    x = np.reshape(x, (4, 5))
    print(x)
    x = np.reshape(x, (10, 2))
    print(x)

    """
    Generate the np array with random values or with specified lower bound and upper bound
    """
    print()
    x = np.random.random((3, 3))  # shape as (3,3)
    print(x)
    # (lower bound, upper bound, shape as(r, c)))
    x = np.random.randint(4, 20, (3, 2))
    print(x)

    """
    Generate the np array with random values in normal distribution
    """
    print()
    x = np.random.normal(0, 0.1, size=(1000, 1000))
    print(x)
    print(f"mean:{x.mean()}")
    print(f"std:{x.std()}")
    print(f"max:{x.max()}")
    print(f"min:{x.min()}")
    print(f"# Positive:", (x > 0).sum())
    print(f"# Negative:", (x < 0).sum())


def numpy_delete_elements_demo():
    x = np.arange(1, 10).reshape(3, 3)
    print(x)

    # delete 1st dimension (axis=0)
    # np.delete(target, index, axis)
    y = np.delete(x, 1, axis=0)
    print('\n', y)

    # delete 2nd dimension (axis=1)
    z = np.delete(x, [0, 2], axis=1)
    print('\n', z)


def numpy_add_elements_demo():
    x = np.array([1, 2, 3, 4, 5])
    print(x)

    x = np.append(x, 6)
    print(x)

    x = np.append(x, [7, 8])
    print(x)

    y = np.arange(1, 10).reshape(3, 3)
    print(y)

    # Append a new ROW
    z = np.append(y, [[10, 11, 12]], axis=0)
    print(z)

    # Append multiple new ROWs
    z2 = np.append(y, [[10, 11, 12], [13, 14, 15]], axis=0)
    print(z2)

    # Append a new column
    z3 = np.append(y, [[10], [11], [12]], axis=1)
    print(z3)


def numpy_insert():
    # insert in column
    x = np.array([1, 2, 5, 6, 10])
    x = np.insert(x, 2, [3, 4])
    print(x)
    print()
    x = np.insert(x, 6, [7, 9])
    print(x)

    print()
    # insert in row
    y = np.array([[1, 2, 3], [7, 8, 9]])
    print(y)
    z = np.insert(y, 1, [4, 5, 6], axis=0)
    print(z)

    # insert in column
    z = np.insert(y, 1, 5, axis=1)
    print(z)


def numpy_stack():
    """
    np.vstack(a, b)  & np.hsatck(a, b)
    these two functions are used to merge two array to one array in the paramater sequence
    Notes: the dimensional should fit for the functions
    """
    x = np.array([[1, 2]])
    print(x)
    print()
    y = np.array([[3, 4], [5, 6]])
    print(y)
    print()

    # vstack
    z = np.vstack((x, y))
    print(z)
    print()
    z2 = np.vstack((y, x))
    print(z2)
    print()

    # hstack
    z3 = np.hstack((y, x.reshape(2, 1)))
    print(z3)
    print()


def numpy_slice_array():
    x = np.arange(1, 21).reshape(4, 5)
    print(x)
    print()

    y = x[1:4, 2: 5]
    print(y)
    print()

    z = x[1:, 2:]
    print(z)
    print()

    z = x[:3, 2:]
    print(z)
    print()

    z = x[:3, 2]
    print(z)
    print()

    z = x[:, 2]
    print(z)
    print()

    z = x[:, 2:3]
    print(z)
    print()

    z = x[:, 2:4]
    print(z)
    print()

    """
    Above only create a 'View' for the array
    when you change the element from the new view, it would modify the original array as well
    """

    print('===================')
    print(x)
    print()
    z = x[1:, 2:]
    z[2, 2] = 666
    print(z)
    print()
    print(x)

    """
    Above only create a 'View' for the array
    when you change the element from the new view, it would modify the original array as well
    If you don't want this happen in your program,
    you have to use np.copy
    """
    print('--------------------')
    x = np.arange(20).reshape(4, 5)
    print(x)
    print()

    z = np.copy(x[1:, 2:])
    # z = (x[1:, 2:]).copy()    # You can use either one for the same result
    print(z)
    print()
    z[2, 2] = 666
    print(z)
    print()
    print(x)

    """
    Build 'index array' as the specific indices you wanna access
    """
    print('===================')
    indices = np.array([1, 3])
    print(indices)

    y = x[indices, :]
    print(y)

    z = x[:, indices]
    print(z)

    """
    get Diagonal elements
    """
    print('===================')
    print(x)
    z = np.diag(x)
    print(z)
    z1 = np.diag(x, k=1)
    print(z1)
    z2 = np.diag(x, k=-1)
    print(z2)


def numpy_filter():
    """
    get Unique elements (sorted) from the array
    """
    print('===================')
    x = np.array([[4, 5, 8], [1, 2, 3], [1, 3, 6]])
    y = np.unique(x)
    print(y)
    print()

    x = np.arange(25).reshape(5, 5)
    print(x)
    print()
    print(x[x > 10])
    print()
    print(x[(x > 10) & (x < 17)])
    print()
    x[(x > 10) & (x < 17)] = -1
    print(x)


def numpy_intersection_union():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 2, 8, 4])
    print(np.intersect1d(x, y))
    print(np.setdiff1d(x, y))
    print(np.union1d(x, y))


def numpy_sort():
    x = np.random.randint(1, 11, size=10)
    print(x)
    print(np.sort(x))
    print(x)


def numpy_computation_operation():
    x = np.array([1, 2, 3, 4, ])
    y = np.array([5, 6, 7, 8])
    print(x)
    print(y)
    print(x + y)
    print(np.add(x, y))
    print(x - y)
    print(np.subtract(x, y))
    print(x * y)
    print(np.multiply(x, y))
    print(x / y)
    print(np.divide(x, y))
    print()

    print(np.sqrt(x))
    print(np.exp(x))
    print(np.power(x, 2))


def numpy_statistic():
    x = np.array([1, 2, 3, 4, ]).reshape(2, 2)
    print(x)
    print("average of all:", x.mean())
    print("average of columns:", x.mean(axis=0))
    print("average of rows:", x.mean(axis=1))
    print("sum of all:", x.sum())
    print("sum of columns:", x.sum(axis=0))
    print("sum of rows:", x.sum(axis=1))
    print("std", x.std())
    print("np.median", np.median(x))
    print("max", x.max())
    print("min", x.min())


def numpy_broadcast_operation_without_loop():
    x = np.array([1, 2, 3, 4, ]).reshape(2, 2)
    print(x)
    print(3 + x)
    print(x - 3)

    print(x * 3)

    print(x / 3)

    print('------------------')
    Y = np.arange(1, 10).reshape(3, 3)
    print(Y)
    X = np.arange(1, 4)
    print(X)
    print(Y + X)
    print(Y - X)
    print(Y * X)
    print(Y / X)


if __name__ == '__main__':
    # test_numpy_speed()

    # np_array_demo()

    # numpy_default_values_demo()

    # numpy_delete_elements_demo()

    # numpy_add_elements_demo()

    # numpy_insert()

    # numpy_stack()

    # numpy_slice_array()

    # numpy_filter()

    # numpy_intersection_union()

    # numpy_sort()

    # numpy_computation_operation()

    # numpy_statistic()

    numpy_broadcast_operation_without_loop()
