from __future__ import division
from mLGHOO import *
import logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
from scipy import stats
import numpy.polynomial.polynomial as poly

#these simulations are Bernoulli variables
arms_def1 = [{
    "name": "x",
    "height_limit": 20,
    "arm_min": 0,
    "arm_max": 1.0,
}]

horizon = 10000

def run_complextrig():
    def complex_trig(input_array):

        x = input_array[0]

        p = 1 / (12 * (np.sin(13 * x) * np.sin(27 * x) + 1))

        reward  = np.random.choice(2, 1, p=[1-p, p])[0]

        return reward

    def complex_trig_no_noise(input_array):
        x = input_array[0]

        p = 1 / (12 * (np.sin(13 * x) * np.sin(27 * x) + 1))

        return p

    np.random.seed(1)
    optObj1 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=1, p_vector=None)
    optObj2 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=5, p_vector=None)



    print "##### Complex trig ####"
    for i in range(horizon):
        arm = optObj1.select_arm()
        reward = complex_trig_no_noise(arm)
        optObj1.update(chosen_arm=arm, reward=reward)

    print 'Result without noise'
    print optObj1.generate_info()

    for i in range(horizon):
        arm = optObj2.select_arm()
        reward = complex_trig(arm)
        optObj2.update(chosen_arm=arm, reward=reward)

    print 'Result with noise'
    print optObj2.generate_info()

def run_uniform():
    def uniform(input_array):

        x = input_array[0]

        p = 0.5

        reward  = np.random.choice(2, 1, p=[1-p, p])[0]

        return reward

    def uniform_no_noise(input_array):
        x = input_array[0]

        p = 0.5

        return p

    np.random.seed(1)
    optObj1 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=1, p_vector=None)
    optObj2 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=5, p_vector=None)



    print "##### Uniform ####"
    for i in range(horizon):
        arm = optObj1.select_arm()
        reward = uniform_no_noise(arm)
        optObj1.update(chosen_arm=arm, reward=reward)

    print 'Result without noise'
    print optObj1.generate_info()

    for i in range(horizon):
        arm = optObj2.select_arm()
        reward = uniform(arm)
        optObj2.update(chosen_arm=arm, reward=reward)

    print 'Result with noise'
    print optObj2.generate_info()


def run_linear():
    def linear(input_array):

        x = input_array[0]

        p = 0.1 + (x-0.1)*0.5

        reward  = np.random.choice(2, 1, p=[1-p, p])[0]

        return reward

    def linear_no_noise(input_array):
        x = input_array[0]

        p = 0.1 + (x-0.1)*0.5

        return p

    np.random.seed(1)
    optObj1 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=1, p_vector=None)
    optObj2 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=5, p_vector=None)



    print "##### Linear ####"
    for i in range(horizon):
        arm = optObj1.select_arm()
        reward = linear_no_noise(arm)
        optObj1.update(chosen_arm=arm, reward=reward)

    print 'Result without noise'
    print optObj1.generate_info()

    for i in range(horizon):
        arm = optObj2.select_arm()
        reward = linear(arm)
        optObj2.update(chosen_arm=arm, reward=reward)

    print 'Result with noise'
    print optObj2.generate_info()


def run_triangle():
    def triangle(input_array):

        x = input_array[0]

        p=[]
        if x < 0.3:
            p = 0.1 + x
        else:
            p = 0.4 - 0.25*x

        reward  = np.random.choice(2, 1, p=[1-p, p])[0]

        return reward

    def triangle_no_noise(input_array):
        x = input_array[0]

        p=[]
        if x < 0.3:
            p = 0.1 + x
        else:
            p = 0.4 - 0.25*x

        return p

    np.random.seed(1)
    optObj1 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=1, p_vector=None)
    optObj2 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=5, p_vector=None)



    print "##### Triangule ####"
    for i in range(horizon):
        arm = optObj1.select_arm()
        reward = triangle_no_noise(arm)
        optObj1.update(chosen_arm=arm, reward=reward)

    print 'Result without noise'
    print optObj1.generate_info()

    for i in range(horizon):
        arm = optObj2.select_arm()
        reward = triangle(arm)
        optObj2.update(chosen_arm=arm, reward=reward)

    print 'Result with noise'
    print optObj2.generate_info()


def run_binormal():
    def binormal(input_array):

        x = input_array[0]

        norm1 = stats.norm(loc=0.4, scale=0.05)
        norm2 = stats.norm(loc=0.8, scale=0.05)
        p = (norm1.pdf(x) / norm1.pdf(0.4) + norm2.pdf(x) / norm2.pdf(0.8))

        reward  = np.random.choice(2, 1, p=[1-p, p])[0]

        return reward

    def binormal_no_noise(input_array):
        x = input_array[0]

        norm1 = stats.norm(loc=0.4, scale=0.05)
        norm2 = stats.norm(loc=0.8, scale=0.05)
        p = (norm1.pdf(x) / norm1.pdf(0.4) + norm2.pdf(x) / norm2.pdf(0.8))

        return p

    np.random.seed(1)
    optObj1 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=1, p_vector=None)
    optObj2 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=5, p_vector=None)



    print "##### Binormal ####"
    for i in range(horizon):
        arm = optObj1.select_arm()
        reward = binormal_no_noise(arm)
        optObj1.update(chosen_arm=arm, reward=reward)

    print 'Result without noise'
    print optObj1.generate_info()

    for i in range(horizon):
        arm = optObj2.select_arm()
        reward = binormal(arm)
        optObj2.update(chosen_arm=arm, reward=reward)

    print 'Result with noise'
    print optObj2.generate_info()



def run_normal():
    def normal(input_array):

        x = input_array[0]

        norm = stats.norm(loc=0.8,scale=0.5)
        p = norm.pdf(x)/norm.pdf(0.8)

        reward  = np.random.choice(2, 1, p=[1-p, p])[0]

        return reward

    def normal_no_noise(input_array):
        x = input_array[0]

        norm = stats.norm(loc=0.8,scale=0.5)
        p = norm.pdf(x)/norm.pdf(0.8)

        return p

    np.random.seed(1)
    optObj1 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=1, p_vector=None)
    optObj2 = mLGHOO(arms_def=arms_def1, v1=10.0, rho=0.5, minimum_grow=5, p_vector=None)


    print "##### normal ####"
    for i in range(horizon):
        arm = optObj1.select_arm()
        reward = normal_no_noise(arm)
        optObj1.update(chosen_arm=arm, reward=reward)

    print 'Result without noise'
    print optObj1.generate_info()

    for i in range(horizon):
        arm = optObj2.select_arm()
        reward = normal(arm)
        optObj2.update(chosen_arm=arm, reward=reward)

    print 'Result with noise'
    print optObj2.generate_info()

if __name__ == '__main__':
    run_complextrig()
    # run_binormal()
    # run_normal()
    # run_linear()
    # run_triangle()
    # run_uniform()
