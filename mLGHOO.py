from __future__ import division
import numpy as np
import copy
import math
import scipy.spatial

import logging
logger = logging.getLogger(__name__)

class mLGHOO():
    def __init__(self, arms_def, v1=1.0, rho=0.5, minimum_grow=1, p_vector=None):
        """
        Initializing a list of arms
       arms_def is the definition of each arm
        """
        #arms_def is a array of dictionary that will define the properties of each dimension
        #here we use the dimension name. It is weird but it would be confusing to use factor


        self.arms_def = arms_def
        if minimum_grow > 0:
            self.minimum_grow = int(minimum_grow)
        else:
            self.minimum_grow = 1
            logger.warning("minimum_grow should be greater than 0")

        if v1 > 0:
            self.v1 = v1  # v1 should be greater than 0
        else:
            self.v1 = 1.0
            logger.warning("v1 should be greater than 0")

        # rho should be < 1 or >0
        if rho < 1 or rho > 0:
            self.rho = rho

        else:
            self.rho = 0.5
            logger.warning("rho should be less than 1")

        if p_vector != None:
            if sum(p_vector) == 1.0:
                self.p_vector = p_vector
            else:
                logger.warning("the probability vector should add to 1")
                self.p_vector = None
        else:
            self.p_vector = None

        self.initial_bound = np.inf
        self.initial_children_index = np.nan
        self.initial_M2 = 0
        self.initial_mean = 0
        self.initial_counts = 0
        self.inital_sumrewards = 0

        #creating list for representing the values for each property by index
        self.number_dimensions = self.get_number_of_dimensions()
        if p_vector!= None and len(p_vector) != self.number_dimensions:
            logger.warning("p_vector should have the same size of the number of dimensions")
            self.p_vector = None

        self.range_size = np.zeros(self.number_dimensions)
        self.midrange_arm = np.zeros(self.number_dimensions)
        self.arm_range_min = np.zeros(self.number_dimensions)
        self.arm_range_max = np.zeros(self.number_dimensions)
        self.height_limit = np.zeros(self.number_dimensions)


        #from 0 to numberdimensions-1
        self.ArmValue = 0
        #Then we go sequentially
        self.Bound = self.ArmValue+self.number_dimensions
        self.Counts =  self.Bound +1
        self.Mean =  self.Counts+1
        self.SumRewards = self.Mean+1
        self.M2 = self.SumRewards+1
        self.Height = self.M2 + 1
        self.ParentIndex =  self.Height + self.number_dimensions
        self.LeftChildren = self.ParentIndex+1
        self.RightChildren = self.LeftChildren+1
        self.DecisionCriteria = self.RightChildren+1
        self.ArmIndex = self.DecisionCriteria + 1



        for i in range(self.number_dimensions):
            self.range_size[i] = self.get_range_size_for_dimensionIndex(i)
            self.midrange_arm[i] = self.get_midrange_arm_for_dimensionIndex(i)
            self.arm_range_min[i] = self.get_min_limit_for_dimensionIndex(i)
            self.arm_range_max[i] = self.get_max_limit_for_dimensionIndex(i)
            self.height_limit[i] = self.get_height_limit_for_dimensionIndex(i)

        initial_arm_value = np.array(self.midrange_arm)
        initial_height = np.ones(self.number_dimensions) #the leave can have different heights for each dimension


        self.arm_list_size = 11 + self.number_dimensions*2
        self.arm_list = np.array([np.zeros(self.arm_list_size)])
        for i in range(self.number_dimensions):
            self.arm_list[0,self.ArmValue+i]=initial_arm_value[i]

        self.arm_list[0,self.Bound]=self.initial_bound
        self.arm_list[0,self.Counts]= 0
        self.arm_list[0,self.Mean]= 0
        self.arm_list[0,self.SumRewards]= 0
        self.arm_list[0,self.M2]= self.initial_M2
        for i in range(self.number_dimensions):
            self.arm_list[0,self.Height+i]= initial_height[i]

        self.arm_list[0,self.ParentIndex]= 0 #the current index
        self.arm_list[0,self.LeftChildren]= self.initial_children_index
        self.arm_list[0,self.RightChildren]= self.initial_children_index
        self.arm_list[0,self.DecisionCriteria] = 0
        self.arm_list[0, self.ArmIndex] = 0



    def debugPrint(self):
        #print self.arm_list
        print('Number of arms: ', self.get_number_of_arms())
        for i in range(self.get_number_of_arms()):
            print('Arm ',i,': ', self.get_arm_value_for_index(i),
                ' Height: ', self.get_heights_for_index(i),
                ' Parent: ', self.get_parent_index_for_index(i),
                ' Mean: ', self.arm_list[i,self.Mean],
                ' Bound: ', self.arm_list[i, self.Bound],
                ' ArmIndex: ', self.get_arm_index_for_index(i),
                ' Children (L,R): ', self.get_left_child_index_for_index(i), self.get_right_child_index_for_index(i))
        #print self.arm_list


 #Some functions of the definition of the problem
    def get_number_of_dimensions(self):
        return len(self.arms_def)
    #get min
    def get_min_limit_for_dimensionIndex(self,index):
        armdic = self.arms_def[index]
        return float(armdic['arm_min'])

    #get max
    def get_max_limit_for_dimensionIndex(self,index):
        armdic = self.arms_def[index]
        return float(armdic['arm_max'])

    #get range_size
    def get_range_size_for_dimensionIndex(self,index):
        armdic = self.arms_def[index]
        arm_min = self.get_min_limit_for_dimensionIndex(index)
        arm_max = self.get_max_limit_for_dimensionIndex(index)
        return np.absolute(arm_max - arm_min)

    #get mid_range_arm
    def get_midrange_arm_for_dimensionIndex(self, index):
        armdic = self.arms_def[index]
        arm_min = self.get_min_limit_for_dimensionIndex(index)
        arm_max = self.get_max_limit_for_dimensionIndex(index)
        return (arm_min + arm_max)/2

    #get height limit
    def get_height_limit_for_dimensionIndex(self,index):
        armdic = self.arms_def[index]
        return float(armdic['height_limit'])

    #get name for index
    def get_name_for_dimensionIndex(self,index):
        armdic = self.arms_def[index]
        return float(armdic['name'])



# Some tree functions
    def has_child_for_index(self,index):
        left_child = self.arm_list[index][self.LeftChildren]
        right_child = self.arm_list[index][self.RightChildren]
        # we want to be sure that both children are nan
        if np.isnan(left_child) and np.isnan(right_child):
            return False
        else:
            return True

    def get_number_of_arms(self):
        return self.arm_list.shape[0]

    def get_heights_for_index(self,index):
        return self.arm_list[index,self.Height:(self.Height+self.number_dimensions)]

    def get_arm_index_for_index(self,index):
        #useful when dealing with subselections of the whole list
        return int(self.arm_list[index, self.ArmIndex])

    def get_arm_value_for_index(self, index):
        return self.arm_list[index, self.ArmValue:(self.ArmValue + self.number_dimensions)]

    def add_arm(self, arm_vector):
        #print 'Before ', self.arm_list
        self.arm_list = np.append(self.arm_list, arm_vector, axis=0)
        #print 'After ',self.arm_list
        return self.get_number_of_arms()-1

## Adding values to the list
    def add_child_for_index(self, index):
        """
        Add a child for in the tree
        :param index:
        :return:
        """

        #to add a child for the tree we need to first select a dimension to divide and we will only divide this dimension
        #Randomly choosing one of the dimensions

        if self.has_child_for_index(index):
            logger.warning("Already have a child")
            return
        else:
            #Here we choose the direction we want to grow according to the p_vector
            # if self.p_vector == None:
            dimension = np.random.choice(self.number_dimensions,1)[0]

            #Checking if we can grow more on this dimension

            current_height = self.arm_list[index,self.Height+dimension]
            limit_height = self.height_limit[dimension]
            if current_height < limit_height:

                #print 'Changing dimension: ', dimension

                current_heights = self.get_heights_for_index(index)
                children_heights = copy.deepcopy(current_heights)
                children_heights[dimension] = children_heights[dimension]+1

                # LeftChildren represents the one that we subtract
                # RightChildren represents the one that we add
                #Arm value for the children
                #It will be the same as the parent for all the rest but the dimension changing
                current_armvalue = self.get_arm_value_for_index(index)
                diff = self.get_range_size_for_dimensionIndex(dimension) / np.power(2,children_heights[dimension])
                #updating only where we divided the space
                left_child_arm = copy.deepcopy(current_armvalue)
                right_child_arm = copy.deepcopy(current_armvalue)
                left_child_arm[dimension] = left_child_arm[dimension] - diff
                right_child_arm[dimension] = right_child_arm[dimension] + diff

                #Left children
                left_children = np.array([np.zeros(self.arm_list_size)])

                for i in range(self.number_dimensions):
                    left_children[0,self.ArmValue+i]=left_child_arm[i]
                left_children[0,self.Bound] = self.initial_bound
                left_children[0,self.Counts] = self.initial_counts
                left_children[0,self.Mean] = self.initial_mean
                left_children[0,self.SumRewards] = self.initial_mean
                left_children[0,self.M2] = self.initial_M2
                #updating the heights
                for i in range(self.number_dimensions):
                    left_children[0,self.Height+i]=children_heights[i]

                left_children[0,self.ParentIndex] = index
                left_children[0,self.LeftChildren] = self.initial_children_index
                left_children[0,self.RightChildren] = self.initial_children_index
                left_children[0,self.DecisionCriteria] =self.initial_bound
                #adding hte arm to the vector
                leftindex = self.add_arm(left_children)
                self.arm_list[leftindex, self.ArmIndex] = leftindex
                self.arm_list[index,self.LeftChildren] = leftindex

                #Right Children
                right_children = np.array([np.zeros(self.arm_list_size)])
                for i in range(self.number_dimensions):
                    right_children[0, self.ArmValue + i] = right_child_arm[i]
                right_children[0, self.Bound] = self.initial_bound
                right_children[0, self.Counts] = self.initial_counts
                right_children[0, self.Mean] = self.initial_mean
                right_children[0, self.SumRewards] =self.initial_mean
                right_children[0, self.M2] = self.initial_M2
                # updating the heights
                for i in range(self.number_dimensions):
                    right_children[0, self.Height + i] = children_heights[i]
                right_children[0, self.ParentIndex] = index
                right_children[0, self.LeftChildren] = self.initial_children_index
                right_children[0, self.RightChildren] = self.initial_children_index
                right_children[0, self.DecisionCriteria] = self.initial_bound
                rightindex = self.add_arm(right_children)
                self.arm_list[rightindex, self.ArmIndex] = rightindex
                self.arm_list[index,self.RightChildren] = rightindex
                return
            else:
                #cannot grow more
                return


    def get_left_child_index_for_index(self, index):
        index = int(index)
        try:
            return int(self.arm_list[index,self.LeftChildren])
        except:
            return -1

    def get_right_child_index_for_index(self, index):
        index = int(index)
        try:
            return int(self.arm_list[index,self.RightChildren])
        except:
            return -1

    def get_root_index(self):
        return 0


    def get_parent_index_for_index(self, index):
        """
        Return the index of the parent given a child index
        :param index:
        :return:
        """
        return int(self.arm_list[index,self.ParentIndex])



    def get_path_for_index(self, index):
        """
        receives an index and returns a numpy array of index for the path
        Searches through the list to find the whole path
        :return:
        """
        list_index = np.array([index])
        node_index = index
        #print node_index
        while node_index!=self.get_root_index():
            node_index = self.get_parent_index_for_index(node_index)
            #print node_index
            list_index = np.append(list_index, node_index)
            #print list_index
        return list_index


    def update_reward_for_index(self, index, sum_rewards):
        self.arm_list[index,self.SumRewards] = self.arm_list[index, self.SumRewards] + sum_rewards

    def update_n_mean_for_index(self, index, n, sum_new_rewards):
        """
        Update the mean and the counts for the played arm
        :param index:
        :param n:
        :param sum_new_rewards:
        :return:
        """
        self.update_reward_for_index(index, sum_new_rewards)

        n_old = copy.deepcopy(self.arm_list[index,self.Counts])
        # update the count for the played arm
        self.arm_list[index, self.Counts] = self.arm_list[index,self.Counts] + n
        n_new = copy.deepcopy(self.arm_list[index,self.Counts])

        old_mean = copy.deepcopy(self.arm_list[index,self.Mean])

        new_mean = copy.deepcopy((old_mean * n_old + sum_new_rewards) / float(n_new))
        self.arm_list[index, self.Mean] = new_mean

    def update_mean_for_index(self, index, reward):
        self.update_n_mean_for_index(index=index, n=1, sum_new_rewards=reward)

    def update_stats_for_index(self,index,reward):
        self.update_M2_for_index(index=index,reward=reward)
        self.update_mean_for_index(index=index,reward=reward)

    # updating the standard deviation iteratively
    def update_M2_for_index(self, index, reward):
        # this is should be called before we update the mean
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        count = self.arm_list[index,self.Counts] + 1
        mean = self.arm_list[index,self.Mean]
        delta = reward - mean
        mean = mean + delta / count
        delta2 = reward - mean
        M2 = self.arm_list[index,self.M2]
        newM2 = M2 + delta * delta2
        self.arm_list[index, self.M2] = newM2

    def get_best_arm_index(self):
        """
        get the highest branch with highest mean value and the highest criteria
        """
        #new algorithm uses the decision criteria, old uses the mean
        best_arm_index = int(self.arm_list[np.argmax(self.arm_list[:, self.DecisionCriteria]), self.ArmIndex])

        return best_arm_index

    def get_total_counts(self):
        return np.sum(self.arm_list[:, self.Counts])

    def calculate_ucb_for_index(self, index):
        """
        Calculate the ucb estimate for an element
        :param index:
        :return:
        """
        counts = self.arm_list[index,self.Counts]

        if counts > 0:
            uncertainty_bound = math.sqrt(
                (2 * math.log(self.get_total_counts())) / float(counts))
            mean = self.arm_list[index,self.Mean]
            h = np.max(self.get_heights_for_index(index))
            max_variation_payoff = self.v1 * np.power(self.rho, h)
            ucb_value = mean + uncertainty_bound + max_variation_payoff
        else:
            ucb_value = np.inf
        return ucb_value

    def calculate_bound_for_index(self, index):
        """
        Calculate the final Bound or B value for the index and save in the bound part of the vector arm_list
        in col index 1
        :param index:
        :return:
        """
        ucb = self.calculate_ucb_for_index(index)
        left_children_bound = np.inf
        right_children_bound = np.inf

        #if it has a child it updates the values
        if self.has_child_for_index(index):
            left_children_index = self.get_left_child_index_for_index(index)
            left_children_bound = self.arm_list[left_children_index,self.Bound]

            right_children_index = self.get_right_child_index_for_index(index)
            right_children_bound = self.arm_list[right_children_index,self.Bound]
        max_bound_children = np.max(np.array([left_children_bound, right_children_bound]))
        bound = np.min(np.array([ucb, max_bound_children]))
        self.arm_list[index,self.Bound] = bound

    def calculate_arm_decision_criteria_for_index(self,index):
        """Decision criteria"""

        #this is better
        bound = self.arm_list[index,self.Bound]
        if np.isnan(bound) or np.isinf(bound):
            self.arm_list[index, self.DecisionCriteria] = 0
        else:
            self.arm_list[index,self.DecisionCriteria] = float(self.arm_list[index,self.Mean])/(float(self.arm_list[index,self.Bound]))


    def update_all_bounds(self):
        """
        In this function we update all bounds
        First we sort by height
        Then we update it sequentially
        we dont calculate for the borders
        :return:
        """
        narms = self.get_number_of_arms()
        for i in reversed(range(narms)):
            self.calculate_bound_for_index(i)
            self.calculate_arm_decision_criteria_for_index(i)


    def extend_tree_for_index(self,index):
        #first we need to check if it actually have enough counts in this node before we can expand it
        current_counts = int(self.arm_list[index,self.Counts])
        if current_counts >= self.minimum_grow:
            self.add_child_for_index(index)
            return
        else:
            return

    def find_nearest_node_by_arm(self,arm):
        #arm is a list with several values e.g. [1.2, 3.4] in the 2D
        #or [1.25] in the 1D
        try:
            #calculating the euclidian distance for all nodes
            err = np.sum((self.arm_list[:,self.ArmValue:(self.ArmValue+self.number_dimensions)]-arm)**2, axis=1)
            #now we have a list wiht hte errors corresponding to the index
            min_index = err.argmin()
            return int(min_index)
        except Exception as e:
            logger.warning("Node not found")
            return -1

    def get_highest_bound_child_for_index(self, index):

        bound_left = []
        bound_right = []

        left_index = self.get_left_child_index_for_index(index)

        right_index = self.get_right_child_index_for_index(index)

        most_promising_child = []
        # if haas left child
        if not np.isnan(left_index):
            bound_left = self.arm_list[left_index,self.Bound]
        else:
            return np.nan
        # if haas right child
        if not np.isnan(right_index):
            bound_right = self.arm_list[right_index,self.Bound]
        else:
            return np.nan

        if bound_right > bound_left:
            most_promising_child = right_index
        if bound_left > bound_right:
            most_promising_child = left_index
        if bound_right == bound_left:
            choose = np.random.choice(2)
            if choose == 1:
                most_promising_child = right_index
            if choose == 0:
                most_promising_child = left_index

        return most_promising_child

    def get_index_path_for_possible_arms(self):
        """We return an array with indexes with arms that we can choose"""
        root_index = self.get_root_index()
        path = []
        arms = []
        arms.append(self.get_arm_value_for_index(root_index))
        path.append(root_index)
        iter_index = root_index
        # while we have children
        children = True
        while children:
            parent_index = iter_index
            iter_index = self.get_highest_bound_child_for_index(parent_index)
            if np.isnan(iter_index) or iter_index==-1:
                children = False
            else:
                path.append(iter_index)
                arms.append(self.get_arm_value_for_index(iter_index))
        return np.array(path)


# ## Algorithm part


    def info(self):
        """
        Info is a dictionary with the info
        :return:
        """
        info = self.generate_info()
        return info

    def select_arm(self,consistency_hash=[],context=[]):
        """
        Selects an arm to play and return the value of the arm
        :return:
        """
        # 1 - Select the path of the most promising children
        path = self.get_index_path_for_possible_arms()

        # Select randomly an arm from the path of the most promising child
        chosen_arm_index = path[np.random.choice(path.size)]
        returnarm = self.get_arm_value_for_index(chosen_arm_index).tolist()

        #Return the value of the arm
        return returnarm


    def update(self, chosen_arm, reward, context=[]):
       #chosen arm is an array of values

        #Extend tree with the new bounds (inf) for the child
        index = self.find_nearest_node_by_arm(chosen_arm)
        #if the chosen arm is not valid we ignore it
        if index == -1:
            logger.warning("no arm selected")
            return
        else:
            # Update for all the path with index
            path = self.get_path_for_index(index)
            for i in path:
                self.update_stats_for_index(index=i, reward=reward)
            self.extend_tree_for_index(index)
            # update statistics for the node
            # Update the computation of all bounds in the tree
            self.update_all_bounds()
            return


    def get_best_arm_value(self):
        armvalue = self.get_arm_value_for_index(self.get_best_arm_index()).tolist()
        return armvalue