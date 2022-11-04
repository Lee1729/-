
import pandas as pd
import numpy as np
from functools import reduce
from collections import Counter
import modules.utils as ut
import modules.visgraph as vg
import modules.splitcriterion as sc
import re

class userTree(object):
    def __init__(self, min_samples, max_depth, params, algorithm='paper', simplify = True):
        self.MIN_SAMPLES = min_samples
        self.MAX_DEPTH = max_depth

        assert algorithm in ['paper', 'adaptive'], 'algorithm = paper or adpative'
        if algorithm == 'paper':
            self.crt = sc.newSplitCrit(self.MIN_SAMPLES, params)
        elif  algorithm == 'adaptive':
            self.crt = sc.adaptiveSplitCrit(self.MIN_SAMPLES, params)

        self.SIMPLIFY = simplify
        self.graph = vg.visGraph()
    #############################################################################################

    def growing_tree(self, data, target_attribute_name, max_path, depth = 1):      
        target_values = data[target_attribute_name]
        cnt_list = ut.count_class(target_values.values, self.NUM_CLASSES)
        leaf_node_class = [np.argmax(cnt_list), target_values]
        
        if(depth > self.MAX_DEPTH) or (len(data)==0) or \
            (len(np.unique(target_values.values)) == 1):         
            return leaf_node_class, self.graph.node_info(cnt_list, self.N_DATA, max_path=max_path, root=False)
        else:       
            self.crt.DEPTH = depth
            [slt_dtype, best_cut, best_feature, left_sub_data, right_sub_data, max_path_srt] = \
                        self.crt.best_split(data, target_attribute_name)           
            if best_feature =='':
                if depth == 1:
                    tree_org = {'Root_node' : leaf_node_class}  
                    leaf_print = {'Root_node' : self.graph.node_info(cnt_list, \
                                                self.N_DATA, max_path='Root', root=True)}
                    return tree_org, leaf_print           
                else:
                    return leaf_node_class, self.graph.node_info(cnt_list, \
                                    self.N_DATA, max_path=max_path, root=False)
            condition = ['<', '>='] if slt_dtype =='n' else ['!=', '==']
            path_cond = [str(max_path_srt =='left'), str(max_path_srt == 'right')]
            
            left_subtree, graph_left_subtree = self.growing_tree(left_sub_data, \
                target_attribute_name, max_path= path_cond[0], depth= depth +1)         
            right_subtree, graph_right_subtree = self.growing_tree(right_sub_data,\
                target_attribute_name, max_path= path_cond[1], depth= depth +1)
            tree = {}
            tree['{} {} {}'.format(best_feature, condition[0], best_cut)] = left_subtree
            tree['{} {} {}'.format(best_feature, condition[1], best_cut)] = right_subtree
            graph_tree = self.graph.get_graph_tree(best_feature, best_cut, cnt_list, condition, \
                                        [graph_left_subtree, graph_right_subtree], max_path)
        return tree, graph_tree

    def recur_simplify(self):
        bf_rule_list = ut.get_leaf_rule(self.tree, [], [], leaf_info=False)
        tree_rule = ut.get_leaf_rule(self.tree, [], [], leaf_info=True)
        print_tree_rule =ut.get_leaf_rule(self.graph_tree, [], [], leaf_info=True)
        
        all_rules = [tuple(i[:-2] + [i[-1][0]]) for i in tree_rule]
        all_print_rules = [tuple(i[:-2] + re.findall('label="(.*)\\\\nhomogeneity' , i[-1])) for i in print_tree_rule]
        all_rules_dict = {tuple(i[:-2] + [i[-1][0]]):i for i in tree_rule}
        all_print_rules_dict = {tuple(i[:-2] + re.findall('label="(.*)\\\\nhomogeneity' , i[-1])):i for i in print_tree_rule}
        dup_rule = [all_rules_dict[r] for r, c in  Counter(all_rules).items() if c >=2]
        dup_print_rule = [all_print_rules_dict[r] for r, c in  Counter(all_print_rules).items() if c >=2]
              
        for n, r in enumerate(dup_rule):
            new_parent_rule = list(r)[:-2] 
            new_parent_print_rule = list(dup_print_rule[n])[:-2]
            org_parent_print_rule = list(dup_print_rule[n])[-2]
            sub_dict = reduce(dict.get, tuple(new_parent_rule), self.tree)
            
            if  isinstance(sub_dict, dict):
                concat_child_df = pd.concat([i[1] for i in sub_dict.values()]) 
                cnt_list  = ut.count_class(concat_child_df.values, self.NUM_CLASSES)
                if len(new_parent_rule) ==0:
                    self.tree = {'Root_node' : [np.argmax(cnt_list), concat_child_df]}
                    leaf_print= self.graph.node_info(cnt_list, self.N_DATA, max_path='Root', root=True)
                    self.graph_tree = leaf_print 
                    return self.tree, self.graph_tree
                else:                  
                    self.tree = ut.setInDict(self.tree , new_parent_rule, [np.argmax(cnt_list),concat_child_df])
                    leaf_print= self.graph.node_info(cnt_list, self.N_DATA, re.findall('max_path = (.*)\"' , org_parent_print_rule)[0], root=False)
                    self.graph_tree = ut.setInDict(self.graph_tree, new_parent_print_rule, leaf_print)
        aft_rule_list = ut.get_leaf_rule(self.tree, [], [], leaf_info=False)
        if bf_rule_list  == aft_rule_list :
            return self.tree, self.graph_tree
        else:
            return self.recur_simplify()
    #############################################################################################
    def fit(self, data, target_attribute_name, depth = 1):
        data = data.copy()
        target_values = data[target_attribute_name].values
        elements = np.unique(target_values)
        self.NUM_CLASSES = len(elements)
        self.CLASS_DICT = {i:v for i, v in enumerate(elements)}
        self.CLASS_DICT_ = {v:i for i, v in enumerate(elements)}
        self.N_DATA = len(data)
        data[target_attribute_name] = [self.CLASS_DICT_[v] for v in target_values]
        
        self.tree, self.graph_tree = self.growing_tree(data, target_attribute_name, max_path='Root', depth=depth)
        if not isinstance(self.tree, dict):
            print('이거뜨면안됨')
            self.tree = {'Root_node' : [1, 1]}
        if list(self.tree.keys())[0] == 'Root_node':
            return self.tree, self.graph_tree
        if self.SIMPLIFY:
            self.recur_simplify()
            
        return self.tree, self.graph_tree
    ############################################################################################
    def predict(self, test, target_attribute_name='target'):
        test = test.copy()
        target_values = test[target_attribute_name]
        test[target_attribute_name] = [self.CLASS_DICT_[v] for v in target_values]
        rule_list = ut.get_leaf_rule(self.tree, [], [], leaf_info=True)    
        predict_class = pd.DataFrame(columns=["class"], index=test.index)
        predict_prob = pd.DataFrame(columns=[str(i) \
                            for i in range(self.NUM_CLASSES)], index=test.index)
        for rule in rule_list:
            idx, pred = ut.recur_split(test, rule, n_class=self.NUM_CLASSES)
            if len(idx)!=0:
                predict_class.loc[idx, 'class'] = [self.CLASS_DICT[np.argmax(pred)]]
                predict_prob.loc[idx, [str(i) \
                    for i in range(self.NUM_CLASSES)]] = [pred] * len(idx)
        return predict_class, predict_prob