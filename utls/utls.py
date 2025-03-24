import time
import numpy as np
import scipy.signal as signal
from collections import Counter
from typing import Iterable

import itertools
from functools import partial
from typing import Callable

from flloat.parser.ltlf import LTLfParser
import flloat

STL_IDS = ["G", "E", "&", "~", "|", "rho"]

class STLNode():
    
    def __init__(self, id: str, children = [], time_bounds: tuple=None, rho=None) -> None:
        self.id = id
        self.children = children
        if self.id == "rho":
            assert len(self.children) == 0
        elif self.id not in ["&", "|", 'True', 'False']:
            # if it's not an 'and' or an 'or', it should only have one child
            assert len(self.children) == 1
        self.rho = rho
        self.time_bounds = time_bounds
        self.order = None
    
    def set_ordering(self, order):
        self.order = order
    
    def __repr__(self) -> str:
        return f'Id:{self.id}, rho:{self.rho}' if self.rho else f'Id:{self.id}, children:{self.children}'
    
    def __str__(self) -> str:
        return f'Id:{self.id}, children:{self.children}'

def parse_stl_into_tree(stl_formula):
    '''
    stl_formula: STL string formatted in MTL library format
    see https://github.com/mvcisback/py-metric-temporal-logic
    env: string
    '''
    # phi = mtl.parse(stl_formula)
    parser = LTLfParser()
    phi = parser(stl_formula.replace("~", "!"))
    root = parse_helpers_ltlf(phi)
    # root = parse_helper(phi)
    return root

    #TODO: include time bounds from the formula if they exist
    # return parse_helper(phi)

def parse_helpers_ltlf(phi):
    #TODO: include time bounds from the formula if they exist
    # match based on id
    
    if str(phi) in ["True", 'true']:
        return STLNode('True') 
    elif str(phi) in ["False", 'false']:
        return STLNode('False') 

    members = phi._members()

    if isinstance(members, str) or len(members) == 1:
        id = "rho"  # at a leaf
        robustness_fxn = str(members)
        return STLNode(id, [], rho=robustness_fxn)
    else:
        id = str(members[0])

    # get the children of the current node
    node_children = []
    try:
        for child in members[1]:
            node_children.append(parse_helpers_ltlf(child))
    except:
        node_children.append(parse_helpers_ltlf(members[1]))

    return STLNode(id, children=node_children)
    
def timeit(func):
    def wrapper(*arg, **kw):
        tic = time.time()
        res = func(*arg, **kw)
        toc = time.time()
        print(func.__name__, toc - tic)
        return res
    return wrapper

def merge_parenthesis(lst):
    fixed, acc = [], []
    total = 0
    for x in lst:
        net = x.count('(') - x.count(')')
        total += net
        acc.append(x)

        if total == 0:
            fixed.append(' & '.join(acc))
            acc = []

    return fixed

def remove_from_str(s:str, tokens_to_rm:Iterable[str]) -> str:
    for t in tokens_to_rm:
        s = s.replace(t, '')
    return s

def binary_to_decimal(X):
    X = np.array(X)
    return X.dot(2**np.arange(X.size)[::-1])

class Episode(object):
    def __init__(self, path, actions, rewards):
        self.path = path
        self.actions = actions
        self.rewards = rewards
    
    def __repr__(self) -> str:
        return "Episode(%s)" % (self.actions)

class Stats(object):
    def __init__(self):
        self.data = []

    def add(self, idx, state):
        floor = state[0].current_floor
        buttons = np.array(state[0].buttons)
        aut_state = state[1]
        buttons = binary_to_decimal(buttons) #buttons.dot(2**np.arange(buttons.size)[::-1])

        self.data.append(tuple([idx, floor, buttons, aut_state]))

    def describe(self):
        return Counter(self.data)

    

def discounted_sum(rewards, discount):
    """
    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=r)
    return y[::-1]


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x, eps=0.01):
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def remap(v, x, y, clip=False):
    if x[1] == x[0]:
        return y[0]
    out = y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])
    if clip:
        out = constrain(out, y[0], y[1])
    return out


def pos(x):
    return np.maximum(x, 0)


def neg(x):
    return np.maximum(-x, 0)


def near_split(x, num_bins=None, size_bins=None):
    """
        Split a number into several bins with near-even distribution.

        You can either set the number of bins, or their size.
        The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def zip_with_singletons(*args):
    """
        Zip lists and singletons by repeating singletons

        Behaves usually for lists and repeat other arguments (including other iterables such as tuples np.array!)
    :param args: arguments to zip x1, x2, .. xn
    :return: zipped tuples (x11, x21, ..., xn1), ... (x1m, x2m, ..., xnm)
    """
    return zip(*(arg if isinstance(arg, list) else itertools.repeat(arg) for arg in args))


def kullback_leibler(p: np.ndarray, q: np.ndarray) -> float:
    """
        KL between two categorical distributions
    :param p: categorical distribution
    :param q: categorical distribution
    :return: KL(p||q)
    """
    kl = 0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi > 0:
                kl += pi * np.log(pi/qi)
            else:
                kl = np.inf
    return kl


def bernoulli_kullback_leibler(p: float, q: float) -> float:
    """
        Compute the Kullback-Leibler divergence of two Bernoulli distributions.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: KL(B(p) || B(q))
    """
    kl1, kl2 = 0, np.infty
    if p > 0:
        if q > 0:
            kl1 = p*np.log(p/q)

    if q < 1:
        if p < 1:
            kl2 = (1 - p) * np.log((1 - p) / (1 - q))
        else:
            kl2 = 0
    return kl1 + kl2


def d_bernoulli_kullback_leibler_dq(p: float, q: float) -> float:
    """
        Compute the partial derivative of the Kullback-Leibler divergence of two Bernoulli distributions.

        With respect to the parameter q of the second distribution.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: dKL/dq(B(p) || B(q))
    """
    return (1 - p) / (1 - q) - p/q


def kl_upper_bound(_sum: float, count: int, threshold: float = 1, eps: float = 1e-2, lower: bool = False) -> float:
    """
        Upper Confidence Bound of the empirical mean built on the Kullback-Leibler divergence.

        The computation involves solving a small convex optimization problem using Newton Iteration

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level
    :param threshold: the maximum kl-divergence * count
    :param eps: Absolute accuracy of the Netwon Iteration
    :param lower: Whether to compute a lower-bound instead of upper-bound
    """
    if count == 0:
        return 0 if lower else 1

    mu = _sum/count
    max_div = threshold/count

    # Solve KL(mu, q) = max_div
    kl = lambda q: bernoulli_kullback_leibler(mu, q) - max_div
    d_kl = lambda q: d_bernoulli_kullback_leibler_dq(mu, q)
    a, b = (0, mu) if lower else (mu, 1)

    return newton_iteration(kl, d_kl, eps, a=a, b=b)


def newton_iteration(f: Callable, df: Callable, eps: float, x0: float = None, a: float = None, b: float = None,
                     weight: float = 0.9, display: bool = False, max_iterations: int = 100) -> float:
    """
        Run Newton Iteration to solve f(x) = 0, with x in [a, b]
    :param f: a function R -> R
    :param df: the function derivative
    :param eps: the desired accuracy
    :param x0: an initial value
    :param a: an optional lower-bound
    :param b: an optional upper-bound
    :param weight: a weight to handle out of bounds events
    :param display: plot the function
    :return: x such that f(x) = 0
    """
    x = np.inf
    if x0 is None:
        x0 = (a + b) / 2
    if a is not None and b is not None and a == b:
        return a
    x_next = x0
    iterations = 0
    while abs(x - x_next) > eps and iterations < max_iterations:
        iterations += 1
        x = x_next

        if display:
            import matplotlib.pyplot as plt
            xx0 = a or x-1
            xx1 = b or x+1
            xx = np.linspace(xx0, xx1, 100)
            yy = np.array(list(map(f, xx)))
            plt.plot(xx, yy)
            plt.axvline(x=x)
            plt.show()

        f_x = f(x)
        try:
            df_x = df(x)
        except ZeroDivisionError:
            df_x = (f_x - f(x-eps))/eps
        if df_x != 0:
            x_next = x - f_x / df_x

        if a is not None and x_next < a:
            x_next = weight * a + (1 - weight) * x
        elif b is not None and x_next > b:
            x_next = weight * b + (1 - weight) * x

    if a is not None and x_next < a:
        x_next = a
    if b is not None and x_next > b:
        x_next = b

    return x_next


def binary_search(f: Callable, eps: float, a: float, b: float = None,
                  display: bool = False, max_iterations: int = 100) -> float:
    """
    Binary search the zero of a non-increasing function.
    :param f: the function
    :param eps: accuracy
    :param a: lower bound for the zero
    :param b: optional upper bound for the zero
    :param display: display the function
    :return: x such that |f(x)| < eps
    """
    x = np.nan
    find_b = False
    if b is None:
        find_b = True
        b = a + 1
    for _ in range(max_iterations):
        x = (a + b) / 2
        f_x = f(x)

        if display:
            import matplotlib.pyplot as plt
            xx0 = a
            xx1 = b
            xx = np.linspace(xx0, xx1, 100)
            yy = np.array(list(map(f, xx)))
            plt.plot(xx, yy)
            plt.axvline(x=x)
            plt.show()

        if f_x > 0:
            a = x
            if find_b:
                b = 2*max(b, 1)
        else:
            b = x
            find_b = False

        if abs(f_x) <= eps:
            break
    else:
        # print("Error: Reached maximum iteration", b)
        pass
    return x


def all_argmax(x: np.ndarray) -> np.ndarray:
    """
    :param x: a set
    :return: the list of indexes of all maximums of x
    """
    m = np.amax(x)
    return np.nonzero(np.isclose(x, m))[0]


def random_argmax(x: np.ndarray) -> int:
    """
        Randomly tie-breaking arg max
    :param x: an array
    :return: a random index among the maximums
    """
    indices = all_argmax(x)
    return np.random.choice(indices)


def random_dist(n):
    q = np.random.random(n)
    return q / q.sum()

