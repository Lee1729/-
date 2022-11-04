
import pandas as pd
import numpy as np
import collections
import graphviz
import ast
import os
from glob import glob
import graphviz
from collections import Counter
from itertools import chain, combinations
import matplotlib.pyplot as plt
import seaborn as sns

class Node:
    def __init__(self, nodeId, label, isRoot=False,parentNode=None,
                 leftNode=None,rightNode=None,isTerminal=False, 
                 attr={}, var_type=None, pred_value=None):
        self.nodeId = nodeId 
        self.label = label 
        self.attr = attr
        self.isRoot = isRoot 
        self.parentNode = parentNode 
        self.leftNode = leftNode 
        self.rightNode = rightNode 
        self.isTerminal = isTerminal 
        self.level = 0 
        self.var_type = var_type
        self.pred_value = pred_value
        
def visualize_tree(tree):
    def add_node_edge(tree, dot=None):
        if dot is None:
            dot = graphviz.Digraph()
#             name = tree
            dot.node(name = str(tree.nodeId), label = str(tree.label), **tree.attr)
        ## left
        if tree.leftNode:
            dot.node(name=str(tree.leftNode.nodeId),label=str(tree.leftNode.label),
                     **tree.leftNode.attr) 
            dot.edge(str(tree.nodeId), str(tree.leftNode.nodeId),
                     **{'taillabel':"yes",'labeldistance':'2.5'})
            dot = add_node_edge(tree.leftNode, dot)
        if tree.rightNode:
            dot.node(name=str(tree.rightNode.nodeId),label=str(tree.rightNode.label),
                     **tree.rightNode.attr)
            dot.edge(str(tree.nodeId), str(tree.rightNode.nodeId),
                    **{'headlabel':" no",'labeldistance':'2'})
            dot = add_node_edge(tree.rightNode, dot)
        return dot
    dot = add_node_edge(tree)
    return dot

def RGBtoHex(vals, rgbtype=1):
    """Converts RGB values in a variety of formats to Hex values.
     @param  vals     An RGB/RGBA tuple
     @param  rgbtype  Valid valus are:
                          1 - Inputs are in the range 0 to 1
                        256 - Inputs are in the range 0 to 255
     @return A hex string in the form '#RRGGBB' or '#RRGGBBAA'
    """
    if len(vals)!=3 and len(vals)!=4:
        raise Exception("RGB or RGBA inputs to RGBtoHex must have three or four elements!")
    if rgbtype!=1 and rgbtype!=256:
        raise Exception("rgbtype must be 1 or 256!")
    #Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
    if rgbtype==1:
        vals = [255*x for x in vals]
    #Ensure values are rounded integers, convert to hex, and concatenate
    return '#' + ''.join(['{:02X}'.format(int(round(x))) for x in vals])

def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

class stableDecisionTree:
    def __init__(self, min_sample = 6, max_depth=100, iter_num=30,
                resample_ratio = 0.7, impurity_measure='entropy', window=5,
                eps = 0.005):
        self.impurity_measure = impurity_measure
        self.root = None
        self.node_id = 0 
        self.col_names = None 
        self.col_types = None 
        self.X = None
        self.y = None
        self.leaf_attr = None 
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.iter_num = iter_num 
        self.resample_ratio = resample_ratio
        self.nodes = {}
        self.rules = None
    
    def traverseInOrder(self, node):
        res = []
        if node.leftNode != None:
            res = res + self.traverseInOrder(node.leftNode)
        res.append(node)
        if node.rightNode != None:
            res = res + self.traverseInOrder(node.rightNode)
        return res
    
    def getDepth(self, root):
        res = self.traverseInOrder(root)
        res = [abs(node.level) for node in res]
        return max(res)
    
    def getLevel(self, node, counter = 1):
        if node.parentNode is None:
            return counter
        else:
            counter += 1
            counter = self.getLevel(node.parentNode,counter)
        return counter
    
    def determineTypeOfCol(self, X, num_unique_values_threshold=1):
        col_types = []
        for col in X.columns:
            unique_values = X[col].unique()
            example_value = unique_values[0]
            if (isinstance(example_value, str)) or (len(unique_values) <= num_unique_values_threshold):
                col_types.append('categorical')
            else:
                col_types.append('continuous')
        self.col_types = col_types

    def isPure(self,y):
        if len(np.unique(y)) > 1:
            return False
        return True
    
    def impurity(self, left_y, right_y):
        y = self.y
        n = len(left_y)+len(right_y)
        pl, pr = len(left_y)/n, len(right_y)/n
        impurity_val = pl*self.individualImpurity(left_y)+\
                        pr*self.individualImpurity(right_y)
        return impurity_val
            
    def individualImpurity(self, y):
        if (self.impurity_measure == 'entropy'):
            return self._entropy(y)
        elif (self.impurity_measure == 'gini'):
            return self._gini(y)
    
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return np.sum([p*(1-p) for p in ps if p > 0])
    def _homogeneity(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return max(ps)
    
    def createLeaf(self, y):
        classes, counts = np.unique(y, return_counts=True)
        index = counts.argmax()
        pred_value = classes[index]
        total_sample = np.sum(counts)
        entropy = self.individualImpurity(y) # optimal 말고 현재 impurity
        entropy = round(entropy,4)
        y_values = [collections.Counter(y)[i] for i in self.y.unique().tolist()]
        question = f'predicted_y = {pred_value}\n'+\
                    f'{self.impurity_measure} : {entropy}\n'+\
                    f'samples : {total_sample}\n'+\
                    f'value : {y_values}'
        return question, pred_value
    
    def powerset_generator(self, i):
        for subset in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
            yield set(subset)
        
    def splitSet(self, x):
        ps = [i for i in self.powerset_generator(x) if i != set() and len(i) != len(x)]
        idx = int(len(ps)/2)
        split_set = []
        for j in range(idx):
            split_set.append(tuple(ps[j]))
        return split_set
        
    def getPotentialSplits(self,X):
        potential_splits = {}
        col_types = self.col_types
        for col_idx in range(X.shape[1]):
            unique_value = np.unique(X[:,col_idx])
            if col_types[col_idx] == 'continuous':
                potential_splits[col_idx] = unique_value
            else:
                potential_splits[col_idx] = unique_value
        return potential_splits
    
    def split(self, X, col_idx, threshold):
        X_col = X[:,col_idx]
        col_types = self.col_types
        if col_types[col_idx] == 'continuous':
            left_idx = np.argwhere(X_col<=threshold).flatten()
            right_idx = np.argwhere(X_col>threshold).flatten()
        else:
            left_idx = np.argwhere(np.isin(X_col,threshold)).flatten()
            right_idx = np.argwhere(~np.isin(X_col,threshold)).flatten()
        return left_idx, right_idx
    
    def determinBestSplit(self, X, y, potential_splits):
        best_split_column, best_split_value, opt_impurity = '', '', ''
        if self.impurity_measure in ['entropy','gini']: 
            opt_impurity = self.individualImpurity(y)
            for col in potential_splits:
                for val in potential_splits[col]:
                    left_idx, right_idx = self.split(X,col,val)
                    if (len(left_idx)<=self.min_sample) or (len(right_idx)<=self.min_sample):
                        continue
                    cur_impurity = self.impurity(y[left_idx],y[right_idx])
                    if cur_impurity < opt_impurity:
                        opt_impurity = cur_impurity
                        best_split_column = col
                        best_split_value = val
        else:
            opt_impurity = -np.infty
            for col in potential_splits:
                for val in potential_splits[col]:
                    left_idx, right_idx = self.split(X,col,val)
                    cur_impurity = self.impurity(y[left_idx],y[right_idx])
                    if cur_impurity >= opt_impurity:
                        opt_impurity = cur_impurity
                        best_split_column = col
                        best_split_value = val        
        return best_split_column, best_split_value, opt_impurity
    
    def mode(self, list_):
        values, counts = np.unique(list_, return_counts=True)
        m = counts.argmax()
        return values[m]
        
    def determinStableSplit(self, X, y):
        var_list=np.array([],dtype='object')
        value_list=np.array([],dtype='object')
        resample_size = int(len(X) * (self.resample_ratio))
        potential_splits = self.getPotentialSplits(X)
        
        for i in range(self.iter_num):
            np.random.seed(i*129)
            resample_idx=np.random.choice(np.arange(len(X)), size=resample_size, replace=False)
            X_train = X[resample_idx]
            y_train = y[resample_idx]
            var, value, opt_impurity =\
            self.determinBestSplit(X_train, y_train, potential_splits)
            if var == '':
                continue
            var_list = np.append(var_list, var)
            value_list = np.append(value_list, value)  
        if len(var_list)==0:
            return None, None
        rep_var = self.mode(var_list[:])
        rep_var_indices = np.where(var_list[:]== rep_var)
        if self.col_types[int(rep_var)] == 'continuous':
            rep_value = np.median(value_list[rep_var_indices].astype(np.float64))
        else:
            rep_value = self.mode(value_list[rep_var_indices])
        return int(rep_var), rep_value
                    
    def fit(self,X,y,
            type_of_col=None, auto_determine_type_of_col=True,
            num_unique_values_threshold = 3):
        '''
        impurity_measure : 불순도 측도
        min_sample : 노드가 포함해야하는 최소 샘플 개수,
        max_depth : 나무 최대 깊이 설정
        type_of_col : 변수 타입 리스트
        auto_determine_type_of_col : 변수 타입 자동 생성 여부
        num_unique_values_threshold : 범주형으로 지정할 최대 유니크 원소 개수
        '''
        self.X = X
        self.y = y
        ### 랜덤으로 칼럼 선택하는 것도 고르자. X = X[random_indices,:]
#         if type_of_col is None:
#             type_of_col = determinTypeOfCol(X)
        if auto_determine_type_of_col:
            self.determineTypeOfCol(X, num_unique_values_threshold)
        else:
            if type_of_col is None:
                raise ValueError('When auto_determine_type_of_col is False, then type_of_col must be specified')
            assert X.shape[1] == len(type_of_col), 'type_of_col has the same length of X columns'
            give_type_of_col = list(set(type_of_col))
            for toc in give_type_of_col:
                if toc != 'categorical' and toc != 'continuous':
                    raise ValueError('type_of_col must contain categorical or continuous')
            self.col_types = type_of_col
        impurity_measures = ['entropy','gini']
        assert self.impurity_measure in impurity_measures,\
                f'impurity_measures must be the one of the {impurity_measures}'
        self.root = self._growTree(X,y)
        
        ### assign node a style
        iod = self.traverseInOrder(self.root)
        root_node = [node for node in iod if node.nodeId == 1][0]
        root_node.isRoot = True
        ## set node level
        for nd in iod:
            nd.level = self.getLevel(nd)
        colors = sns.color_palette('hls', self.getDepth(self.root))
        ## set node level
        leaf_color = sns.color_palette('pastel', len(np.unique(y)))
#             class_to_color = dict()
#             for i, l in enumerate(np.unique(y)):
#                 class_to_color[l] = RGBtoHex(leaf_color[i])
        leaf_attr = dict()
        for i, l in enumerate(np.unique(y)):
            leaf_attr[l] = {'shape':'box', 'color':f'{RGBtoHex(leaf_color[i])}', 
                                   'fontcolor':f'{RGBtoHex(leaf_color[i])}','peripheries':'2'}
        self.leaf_attr = leaf_attr
        for l in range(1,self.getDepth(self.root)+1):
            color = RGBtoHex(colors[l-1])
            for nd in iod:
                if nd.level == l:
                    if nd.isTerminal:
                        nd.attr = {'shape':'box',
                                   'peripheries':'2'}
                    else:
                        nd.attr = {'shape':'box'}
                        
    def _growTree(self, X, y, counter=0): ## Tree 배고 노드 클래스만 가지고 해야겠다.
        self.node_id += 1
        if counter == 0:
            global col_names
#             col_types = self.col_types
            col_names = X.columns
            self.col_names = X.columns
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
        else:
            X = X
        if (self.isPure(y)) or (len(y) <= self.min_sample) or (counter == self.max_depth):
            leaf, pred_y = self.createLeaf(y)
            if isinstance(leaf, float):
                if not leaf.is_integer():
                    leaf = round(leaf,2)
            node = Node(self.node_id, label=leaf, pred_value=pred_y, isTerminal=True)
            self.nodes[self.node_id] = node
            return node
        else:
            counter += 1
            best_split_column, best_split_value =\
                self.determinStableSplit(X=X, y=y)
            if best_split_column == None:
                leaf, pred_y = self.createLeaf(y)
                if isinstance(leaf, float):
                    if not leaf.is_integer():
                        leaf = round(leaf,2)
                node = Node(self.node_id, label=leaf, pred_value=pred_y, isTerminal=True)
                self.nodes[self.node_id] = node
                return node
            
            left_idx, right_idx = self.split(X,best_split_column,best_split_value)
            if (len(left_idx) <= self.min_sample) or (len(right_idx) <= self.min_sample):
                leaf, pred_y = self.createLeaf(y)
                if isinstance(leaf, float):
                    if not leaf.is_integer():
                        leaf = round(leaf,2)
                node = Node(self.node_id, label=leaf, pred_value=pred_y, isTerminal=True)
                self.nodes[self.node_id] = node
                return node
            
            entropy = self.individualImpurity(y) # optimal 말고 현재 impurity
            entropy = round(entropy,4)
            total_sample = len(y)
            y_values = [collections.Counter(y)[i] for i in collections.Counter(y).keys()]
            col_name = col_names[best_split_column]
            if self.col_types[best_split_column] == 'continuous':
                question = f'{col_name} <= {round(best_split_value,4)}\n'+\
                            f'{self.impurity_measure} : {entropy}\n'+\
                            f'samples : {total_sample}\n'+\
                            f'value : {y_values}'
            else:
                question = f'{col_name} == {best_split_value}\n'+\
                            f'{self.impurity_measure} : {entropy}\n'+\
                            f'samples : {total_sample}\n'+\
                            f'value : {y_values}'
#             sub_tree = {question:[]}
            node = Node(self.node_id, label=question, 
                        var_type = self.col_types[best_split_column])
            self.nodes[self.node_id] = node

            left_child = self._growTree(X[left_idx,:],y[left_idx],counter)  
            right_child = self._growTree(X[right_idx,:],y[right_idx],counter)
            if left_child.label == right_child.label:
                node = left_child
            else:
                node.leftNode = left_child
                node.rightNode = right_child
                left_child.parentNode = node
                right_child.parentNode = node

            return node
    
    def predict(self,X):
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])
    
    def _traverse_tree(self, x, node):
        if node.isTerminal:
            return node.pred_value

        question = node.label.split('\n')[0]
        
        if ' <= ' in question:
            col_name, value = question.split(' <= ')
            if x[col_name] <= float(value):
                return self._traverse_tree(x, node.leftNode)
            return self._traverse_tree(x, node.rightNode)
        else:
            col_name, value = question.split(' == ')
            if x[col_name] == value:
                return self._traverse_tree(x, node.leftNode)
            return self._traverse_tree(x, node.rightNode)

    def pruning(self, node, X_val, y_val):
        X = self.X
        y = self.y
        if isinstance(y, pd.Series):
            y = y.values
        return self._pruning(node, X, y, X_val, y_val)
    
    def _filterX(self, X, node):
        question = node.label.split('\n')[0]
        if ' <= ' in question:
            col_name, value = question.split(' <= ')
            yes_index = X.loc[X[col_name] <= float(value)].index
            no_index = X.loc[X[col_name] > float(value)].index
        else:
            col_name, value = question.split(' == ')
            yes_index = X.loc[X[col_name].isin(ast.literal_eval(value))].index
            no_index = X.loc[~X[col_name].isin(ast.literal_eval(value))].index
        return yes_index, no_index
    
    def _pruning_leaf(self, node, X, y, X_val, y_val):
        classes, counts = np.unique(y, return_counts=True)
        index = counts.argmax()
        leaf = classes[index]
        errors_leaf = np.sum(y_val != leaf)
        errors_decision_node = np.sum(y_val != self.predict(X_val)) ##<---self로 바꿔야해
        if errors_leaf <= errors_decision_node:
            if isinstance(leaf, float):
                if not leaf.is_integer():
                    leaf = round(leaf,2)
            return Node(node.nodeId, label=leaf, isTerminal=True,
                        attr=self.leaf_attr[leaf])
        else:
            return node
        
    def _pruning(self, node, X, y, X_val, y_val):
#         assert self.root is not None, 'you must fit first'
        X = X.reset_index(drop=True)
        left_child = node.leftNode
        right_child = node.rightNode
        if node.leftNode.isTerminal == True and node.rightNode.isTerminal == True:
            return self._pruning_leaf(node, X, y, X_val, y_val)
        else:
            tr_yes_idx, tr_no_idx = self._filterX(X, node)
            val_yes_idx, val_no_idx = self._filterX(X_val, node)

            if node.leftNode.isTerminal == False:
                left_child = self._pruning(node.leftNode, X.loc[tr_yes_idx], y[tr_yes_idx],
                              X_val.loc[val_yes_idx], y_val[val_yes_idx])
#                     node.leftNode = left_child
#                     left_child.parentNode = node
            if node.rightNode.isTerminal == False:
                right_child = self._pruning(node.rightNode, X.loc[tr_no_idx], y[tr_no_idx],
                              X_val.loc[val_no_idx], y_val[val_no_idx])

#                     node.rightNode = right_child
#                     right_child.parentNode = node
#                 question = node.label.split('\n')[0]
            attr = node.attr
            node = Node(node.nodeId, label=node.label, isTerminal=node.isTerminal)
            node.attr = attr
            node.leftNode = left_child
            left_child.parentNode = node
            node.rightNode = right_child
            right_child.parentNode = node
        return self._pruning_leaf(node, X, y, X_val, y_val)
    
    def export_text(self,):
        paths = []
        path = []
        def _traverse_tree(node,path,paths):
            p1, p2 = list(path), list(path)
            if node.leftNode:
                question = node.label.split('\n')[0]
                p1 += [f"{question}"]
                _traverse_tree(node.leftNode,p1,paths)
            if node.rightNode:
                question = node.label.split('\n')[0]
                if ' <= ' in question:
                    col_name, value = question.split(' <= ')
                    p2 += [f"{col_name} > {value}"]
                else:
                    col_name, value = question.split(' == ')
                    p2 += [f"{col_name} != {value}"]
                _traverse_tree(node.rightNode,p2,paths)
            if node.isTerminal==True:
                path += [(node.pred_value)]
                paths += [path]

        _traverse_tree(self.nodes[1],path,paths)

        # sort by samples count
        samples_count = [p[-1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]
        self.rules = paths

        rules = []
        for path in paths:
            rule = "IF "
            for p in path[:-1]:
                if rule != "IF ":
                    rule += " AND "
                rule += str(p)
            rule += " THEN "
            rule += "y = "+str(path[-1])
            #rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]
        for r in rules:
            print(r)