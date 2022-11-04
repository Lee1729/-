# 주) refactoring이 전혀 되지 않았음. 코드가 매우 더러움 _lee

import pandas as pd
import numpy as np
from functools import reduce
import collections
import modules.utils as ut
from modules.usertree import userTree as utr
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import graphviz
import re

class stable_concise_rule_induction(object):
    def __init__(self, min_samples=6, max_depth=100, algorithm='adaptive', simplify = True):
        '''
        MIN_SAMPLES:  종료 기준, 분기 전 시점 최소 샘플 수
        MAX_DEPTH: 논문에 나와있는 조건을 만족하는 depth
        ALGORITHM: 'paper' or 'adaptive'
        SIMPLIFY: 자식노드가 부모노드의 예측값과 같을 때 합칠 것인지 말 것인지 결정
        params: paper version - lambda값, adative version - lambda 범위 값(array형태)
        '''
        self.MIN_SAMPLES = min_samples
        self.MAX_DEPTH = max_depth
        self.ALGORITHM = algorithm
        self.SIMPLIFY = simplify
        self.params = sorted([1-np.log10(i) for i in np.arange(1,10,1)])
        
    ############################################################################################
    def concise_Rule(self, train_list, rule_dic_list, z_a=-1.645):
        
        def con(rule, max_path, subset):
            cond = rule.split()[1]  
            att, value = rule.split(' '+cond+' ')
            if cond == '>=':
                if max_path:
                    cond_adv = '>='
                    return subset.loc[subset[att] >= float(value),:].index
                else:
                    cond_adv = '<'
                    return subset.loc[subset[att] < float(value),:].index
            elif cond == '==':
                if max_path:
                    cond_adv = '=='
                    return subset.loc[subset[att] == value,:].index
                else:
                    cond_adv = '!='
                    return subset.loc[subset[att] != value,:].index
        n=1
        z_a=z_a
        target_values = train_list[0][self.target_name].unique()
        target_values.sort()
        list_df_concise_rule_info = []
        concised_rule_dic_list = []

        for df, rule_dic_ex in zip(train_list, rule_dic_list):
            TS ={}
            S = {}
            TS[0]=df
            homogeneity = []
            coverage = []
            condition_num = [] 
            rule_length_list = [] 
            pred_class = []
            trainset_size = df.shape[0]
            left_subset = df
            right_subset = df
            present_subset = df
            rep = len(rule_dic_ex.keys()) -1

            for rule_nm in range(rep):
                subset = present_subset
                subset_z = present_subset
                test_num = 0
                df_total = pd.DataFrame()
                rule_list = rule_dic_ex[rule_nm][:-1]
                rule_length = len(rule_dic_ex[rule_nm][:-1])
                rule_length_list.append(rule_length)
                pred_class.append(rule_dic_ex[rule_nm][-1][0])
                for i in range(rule_length):
                    rule_list_temp = rule_list.pop(0)   
                    for rule, max_path in [rule_list_temp]:      
                        subset = subset.loc[con(rule, max_path, subset),:]
                left_subset = present_subset.loc[subset.index,:]
                right_subset = present_subset.loc[present_subset.index.difference(subset.index),:]
                present_subset = right_subset      
                TS[rule_nm+1]=present_subset
                S[rule_nm]=subset
                
            S[rule_nm+1] = present_subset.loc[present_subset.index.difference(S[rule_nm].index),:]
            rule_length_list.append(rule_length_list[-1])
            pr = []
            
            for j in range(1, rep+1):
                prob = max((S[j][self.target_name]==values).sum() for values in target_values) / (S[j].shape[0]+0.0000001)
                pr.append(prob)
                if pr[j-1] == 1:
                    pr[j-1] = 0.999
            pr_up = []
            n_samples = []

            for j in range(1, rep):
                pr_up.append([])
                n_samples.append([])
                if j==rep-1:
                    pr_up.append([])
                    n_samples.append([])
                for i in range(j):     
                    subset=TS[i]
                    rule_list = rule_dic_ex[j][:-1]
                    rule_length = len(rule_dic_ex[j][:-1]) 
                    for c in range(rule_length):
                        rule_list_temp = rule_list.pop(0)   
                        for rule, max_path in [rule_list_temp]:     
                            subset = subset.loc[con(rule, max_path, subset),:]

                    prob = max((subset[self.target_name]==values).sum() for values in target_values) / (subset.shape[0]+0.00001)
                    pr_up[j-1].append(prob)
                    n_samples[j-1].append(subset.shape[0])
                    if j==rep-1:
                        adv_subset = TS[i].loc[TS[i].index.difference(subset.index),:]
                        prob = max((adv_subset[self.target_name]==values).sum() for values in target_values) / (adv_subset.shape[0]+0.00001)
                        pr_up[j].append(prob)
                        n_samples[j].append(adv_subset.shape[0])     
            z = []
            j_star=[]

            for j in range(len(pr_up)):
                z.append([])
                if j == len(pr_up) -1 :
                    idx=[j]
                else:
                    idx=[j+1]
                for i in range(0, len(pr_up[j])):
                    z_ij = (pr_up[j][i] - pr[j]) / np.sqrt((pr[j] * (1 - pr[j]) +0.000001 ) / (n_samples[j][i]+0.000001))
                    z[j].append(z_ij)
                    if z[j][i] > z_a:
                        idx.append(i)
                i_min = min(idx)
                j_star.append(i_min)     
                if j < len(pr_up)-1:
                    if i_min <= j : 
                        homogeneity.append(pr_up[j][i_min])
                        coverage.append(n_samples[j][i_min] / trainset_size)
                    else : 
                        homogeneity.append(pr[j])
                        coverage.append(S[j].shape[0] / trainset_size)
                else:
                    if i_min < j : 
                        homogeneity.append(pr_up[j][i_min])
                        coverage.append(n_samples[j][i_min] / trainset_size)
                    else: 
                        homogeneity.append(pr[j])
                        coverage.append(S[j].shape[0] / trainset_size)           
                        
            j_star.insert(0, 0)
            homogeneity.insert(0, max((S[0][self.target_name]==values).sum() for values in target_values) / (S[0].shape[0]+0.000001))
            coverage.insert(0, S[0].shape[0]/trainset_size)  
            condition_num = rule_length_list.copy()
            i=0

            for j in j_star :
                rule_length = rule_length_list[i] 
                if j == 0 :
                    condition_num[i] = rule_length
                elif j > 0:
                    condition_num[i] = rule_length + sum(rule_length_list[0:j])
                i += 1
            lst = range(len(z))
            
            if len(z) ==0:
                lst = [1]
            col_name = ["Rule{:}".format(x) for x in lst]
            col_name.append("adv_Rule{:}".format(lst[-1]))
            index_name = ['pred', 'j*', 'Homogeneity', 'Coverage', '# of condition']
            df_concise_rule_info = pd.DataFrame([pred_class, j_star, homogeneity, coverage, condition_num], 
                                                index=index_name[0:5], columns=col_name)
            df_concise_rule_info.name = "%d번째"%n
            list_df_concise_rule_info.append(df_concise_rule_info)
            n+=1

        return list_df_concise_rule_info        
    
    ############################################################################################
    def concised_rule_extraction(self, rule_dic_list, rule_info):
        def refine_rule(rule): 
            refined_rule =[]
            for r in rule:
                condition, max_path = r
                sign = condition.split()[1]
                var, value = condition.split(' '+sign+' ')
                if max_path==False:
                    if sign == '>=': 
                        sign = '<'
                    elif sign == '==':
                        sign = '!='
                refined_condition = ('{} {} {}').format(var,sign,value)
                refined_rule.append(refined_condition)
            return refined_rule

        def adverse_rule(rule): 
            adv_rule =[]
            for condition in rule:
                sign = condition.split()[1]
                var, value = condition.split(' '+sign+' ') 
                if sign == '>=': 
                    sign = '<'
                elif sign == '==':
                    sign = '!='
                elif sign == '<':
                    sign = '>='
                elif sign == '!=':
                    sign = '=='
                adv_condition = ('{} {} {}').format(var,sign,value)
                adv_rule.append(adv_condition)
            return adv_rule

        def rule_tostring(concised_rule):
            str_rule=''
            for rule in concised_rule:
                if rule == concised_rule[-1]:
                    a_str_rule = rule
                else: a_str_rule = rule+', '
                str_rule += a_str_rule
            return str_rule 
    
        concised_rule_dic_list = {}
        df_concise_rule = {}
        tree_num=0

        for rule_dic_ex in rule_dic_list:
            rule_info[tree_num] = rule_info[tree_num].fillna(0)
            j_star = list(rule_info[tree_num].loc['j*'].astype(int).values)
            concised_rule_list={}
            rulestring_list=[]
            rep = len(rule_dic_ex.keys()) -1   
            for rule_nm in range(0,rep):
                rule = refine_rule(rule_dic_ex[rule_nm][:-1])
                j = j_star[rule_nm]
                str_rule = ''
                if j!=0:
                    for i in range(j-1, -1,-1):
                        split_rule = refine_rule(rule_dic_ex[i][:-1])
                        split_condition = adverse_rule(split_rule)
                        for r in split_condition[::-1]:
                            rule.insert(0,r)
                concised_rule_list[rule_nm] = rule
                rulestring_list.append(rule_tostring(rule))

                rule.append((int(rule_info[tree_num].loc['pred'][rule_nm]), 
                   rule_info[tree_num].loc['Homogeneity'][rule_nm].round(4),
                   rule_info[tree_num].loc['Coverage'][rule_nm].round(4)))
            df_rules = pd.DataFrame([rulestring_list, rule_info[tree_num].loc['pred'][:-1].astype(int), 
                                         rule_info[tree_num].loc['Homogeneity'][:-1].round(4),
                                         rule_info[tree_num].loc['Coverage'][:-1].round(4),
                                         rule_info[tree_num].loc['# of condition'][:-1]], 
                                         index=['Rule', 'pred_y', 'Homogeneity', 'Coverage', '# of condition'])
            concised_rule_dic_list[tree_num] = concised_rule_list
            df_concise_rule[tree_num] = df_rules
            tree_num += 1

        return concised_rule_dic_list, df_concise_rule

    def sort_homo(self, df_rule):
        df_rule = df_rule.T.sort_values(by=['Homogeneity'], ascending=False) 
        df_rule = df_rule.reset_index(drop=True)
        return df_rule.T        
    
    def multi_to_integer(self,y):
        target_values = y.values
        elements = np.unique(target_values)
        CLASS_DICT = {v:i for i, v in enumerate(elements)}
        y_int = [CLASS_DICT[v] for v in target_values]
        return y_int
    
    def comp(self,var, sign, value, subset):
        if sign == '>=':
            return subset.loc[subset[var] >= float(value),:].index
        elif sign == '<':
            return subset.loc[subset[var] < float(value),:].index
        elif sign == '==':
            return subset.loc[subset[var] == value,:].index
        elif sign == '!=':
            return subset.loc[subset[var] != value,:].index

    ############################################################################################
    def fit(self, data, target_name='target', output_graph = False, save_dir='save_dir', d_set='datanm', rule_rate=0.9,
           iter_num=20, resample_ratio=0.85):
        df = data
        df_org = data
        self.target_name = target_name
        elements = np.unique(df[self.target_name].values)
        self.NUM_CLASSES = len(elements)
        self.CLASS_DICT = {i:v for i, v in enumerate(elements)}
        self.CLASS_DICT_ = {v:i for i, v in enumerate(elements)}
        self.N_DATA = len(df_org)
        self.df_total_inf_base = pd.DataFrame()
        self.rule_model= {}
        self.analy_sample = {}
        self.rule_tra_ind_num = 0
        self.others_df = pd.DataFrame()
        self.rule_rate = rule_rate
        self.concised_rule = []
        self.df_concise_rule = pd.DataFrame()
        self.rf=None
        self.resample_ratio = resample_ratio
        self.iter_num = iter_num
        rule_idx = 0
        rule_idx_ = 1
        used_idx = [] 
        train_data = df_org.copy()
        rule_info = pd.DataFrame()
        rule_gen_terminate = int(data.shape[0]*(1-self.rule_rate))
        
        while True:     
            not_used_idx = list(set(df.index) - set(used_idx))
            if len(not_used_idx) <= rule_gen_terminate:
                break
            df = df.loc[not_used_idx, ]
            trees=[]
            graph_trees=[]
            resample_size = int(len(df) * (self.resample_ratio))

            tree_ins = utr(self.MIN_SAMPLES, self.MAX_DEPTH, params=self.params, \
                           algorithm=self.ALGORITHM, simplify = self.SIMPLIFY)     
            for i in range(self.iter_num):
                np.random.seed(i*129)
                resample_idx=(np.random.choice(np.arange(len(df)), size=resample_size, replace=True))
                resample_data = df.iloc[resample_idx]
                tree, graph_tree = tree_ins.fit(resample_data, target_attribute_name = self.target_name)
                trees.append(tree)
                graph_trees.append(graph_tree)
                
            boot_rule_list = []
            boot_tr = []
            for tree, graph_tree in zip(trees, graph_trees):
                if list(tree.keys())[0] == 'Root_node':                
                    target_values= np.array([self.CLASS_DICT_[v] for v in df[self.target_name].values])
                    cnt_list = np.array(ut.count_class(target_values, self.NUM_CLASSES))
                    pred_prob = cnt_list/np.sum(cnt_list)
                    self.rule_model[-1] = [(np.argmax(pred_prob), pred_prob, len(not_used_idx)/self.N_DATA)]
                    continue
                tree_rule = ut.get_leaf_rule(tree, [], [], leaf_info=True)
                graph_tree_rule = ut.get_leaf_rule(graph_tree, [], [], leaf_info=True)            
                each_rule_list =[]
                class_number = {}
                for i in elements:    
                    class_number[i] = len(df_org[df_org[self.target_name] == i])
                cnt_list_df = list(class_number.values())
                cnt_list_df = list(map(int, cnt_list_df))
                for tr_list, gtr_list in zip(tree_rule, graph_tree_rule):   
                    temp=[]
                    each_rule_list=[]
                    max_path_list = []
                    for tr, gtr in zip(tr_list[:-1], gtr_list[1:]):
                        mxp_str = re.findall('max_path = (.*)\"' , gtr)[0]
                        max_path = True if mxp_str in ['True', 'Root'] else False
                        max_path_list.append(max_path)
                        temp.append((max_path,mxp_str ))               
                        if not max_path:
                            each_rule_list  = []
                            break
                        direction = False if tr.split()[1] in ['<', '!='] else True
                        tr = tr.replace('<', '>=').replace('!=', '==')
                        each_rule_list.append((tr, direction))
                    if len(max_path_list) == sum(max_path_list):  
                        boot_rule_list.append(each_rule_list)
                        boot_tr.append(tr_list)
            if len(boot_rule_list) == 0:
                break
            if len(boot_rule_list) < int(iter_num/1.7):
                break
            tuple_lst = [tuple(l) for l in boot_rule_list]
            counts = collections.Counter(tuple_lst)
            rep_rule = counts.most_common(1)[0][0][:]
            for rep_id in range(len(tuple_lst)):
                if tuple_lst[rep_id] == rep_rule:
                    break   
            rep_rule = boot_rule_list[rep_id]
            tree = trees[rep_id]
            if list(tree.keys())[0] == 'Root_node':                
                target_values= np.array([self.CLASS_DICT_[v] for v in df[self.target_name].values])
                cnt_list = np.array(ut.count_class(target_values, self.NUM_CLASSES))
                pred_prob = cnt_list/np.sum(cnt_list)
                self.rule_model[-1] = [(np.argmax(pred_prob), pred_prob, len(not_used_idx)/self.N_DATA)]
                break
            
            graph_tree = graph_trees[rep_id]
            tree_rule = ut.get_leaf_rule(tree, [], [], leaf_info=True)
            graph_tree_rule = ut.get_leaf_rule(graph_tree, [], [], leaf_info=True)            
            each_rule_list =[]
            class_number = {}
            for i in elements:    
                class_number[i] = len(df_org[df_org[self.target_name] == i])
            cnt_list_df = list(class_number.values())
            cnt_list_df = list(map(int, cnt_list_df))
            
            if len(df_org) == len(not_used_idx):                
                n_data=len(df_org)                
                class_prior = [cnt_list_df[i]/n_data for i in range(len(cnt_list_df))]
                dic = dict.fromkeys(range(0, len(graph_tree_rule)), [])
                for num, graph in enumerate(graph_tree_rule):
                    lists = []
                    for graph_sp in graph[:-1]:                        
                        testtest1 = graph_sp.replace('[', '').replace(']', '').replace('"', '')
                        te1= testtest1.split('\\')[0]
                        te1 = te1.replace('label=', '')
                        lists.append(te1)
                    dic[num] = lists
                    leaf_info = graph[-1].replace('[', '').replace(']', '').replace('"', '')    
                    leaf_info = leaf_info.split('\\')                    
                    attr_num_ = len(lists)
                    pred_ = leaf_info[0].replace('label=predict =', '')
                    pred_ = int(pred_)                    
                    homogeneity_ = leaf_info[1].replace('nhomogeneity =', '')
                    homogeneity_ = float(homogeneity_)         
                    coverage_ = leaf_info[2].replace('ncoverage =', '')
                    coverage_ = float(coverage_)         
                    lift_ = np.round(homogeneity_ / class_prior[pred_], 4)
                    sort_col = ['pred', 'depth', 'homogeneity', 'lift', \
                                'coverage', 'split_criterion']
                    sort_ind = [f'rule_{num}']  
                    datas = \
                    {sort_ind[0] : [pred_, attr_num_, homogeneity_, lift_, \
                                coverage_, dic[num]]}  
                    df_total = pd.DataFrame(datas, index=sort_col)
                    rule_info = pd.concat([rule_info, df_total], axis=1)
            subset=df.copy()
            for cond in rep_rule:
                ops = cond[0].split()[1]
                var, value = cond[0].split(' '+ops+' ')
                if cond[1] == True:
                    if ops == '==':
                        subset=subset[subset[var]==value]
                    elif ops == '>=':
                        subset=subset[subset[var]>=float(value)]
                else: 
                    if ops == '==':
                        subset=subset[subset[var]!=value]
                    elif ops == '>=':
                        subset=subset[subset[var]<float(value)]
                        
            used_idx=subset.index
            cnt_list = np.array(ut.count_class(boot_tr[rep_id][-1][-1].values, self.NUM_CLASSES))
            pred_prob = cnt_list/(sum(cnt_list))
            homogeneity = np.round(max(cnt_list)/sum(cnt_list),3)
            train_index = len(used_idx)

            if np.abs(rule_idx - rule_idx_) == 1 or rule_idx == 0:       
                if self.rule_tra_ind_num < np.round(self.N_DATA * rule_rate):
                    rep_rule.append((boot_tr[rep_id][-1][0], pred_prob, round(sum(cnt_list)/self.N_DATA, 3)))
                    self.rule_model[rule_idx] = rep_rule             
                    rule_idx_ += 1
                    homo_val = pd.DataFrame(data=homogeneity, columns = ["rule_%d"%rule_idx],  index = ['Homogeneity'])
                    tra_ind = pd.DataFrame(data=train_index, columns = ["rule_%d"%rule_idx],  index = 
                                           ['The_number_of_train_index'])         
                    self.rule_tra_ind_num += train_index                
                    data_frames = [homo_val, tra_ind.astype(float)]
                    df_merged = reduce(lambda left,right: pd.merge(left,right,how='outer', on="rule_%d"%rule_idx), data_frames)
                    df_merged.index =['Homogeneity', 'The_number_of_train_index']
                    self.df_total_inf_base = pd.concat([self.df_total_inf_base, df_merged], axis=1)
                    rule_label = boot_tr[rep_id][-1][-1]
                    temp_df = train_data[np.in1d(train_data.index, rule_label.index) == True]         
            else:
                rule_idx_ += 100
            rule_idx +=1
            del tree_ins                
            
        rule_rule = pd.DataFrame(index=['Rule_predict','The_number_of_rule_attribute','Train_coverage', 
                                        'Train_cumulative_coverage'])
        cov_cum_list = {}
        
        for key, value in self.rule_model.items():
            if key != -1:
                val = float(value[-1][2])
                cov_cum_list[key] = val
                cov_cum = sum(list(cov_cum_list.values()))
                rule_rule.loc['Rule_predict','rule_%d'%key] = value[-1][0]
                rule_rule.loc['The_number_of_rule_attribute','rule_%d'%key] = len(value[:-1])
                rule_rule.loc['Train_coverage','rule_%d'%key] = value[-1][2]
                rule_rule.loc['Train_cumulative_coverage','rule_%d'%key] = cov_cum  
                
        self.df_total_inf_base = pd.concat([self.df_total_inf_base, rule_rule], axis=0)
        
        data = train_data.copy()
        colnm = data.columns
        X = train_data.loc[:,colnm [colnm != self.target_name]]
        y = train_data.loc[:, self.target_name]
        in_feature = list(data.columns [data.columns != self.target_name])
        cate_col = [col for col in in_feature \
                    if not np.issubdtype(X[col ].dtype, np.number)]
        X_dummies = pd.get_dummies(data.loc[:,in_feature], columns=cate_col)       
        
        y_train_d = train_data.loc[:, self.target_name]
        y_train_d.loc[:] = self.multi_to_integer(y_train_d)
        y_train_d = y_train_d.astype('int64')

        rf = RandomForestClassifier(n_estimators=55, random_state=3111)
        rf.fit(X_dummies,y_train_d)
        self.rf = rf

        df_tree_dic_all = {}
        train_list = [train_data]
        rule_dic_list = [self.rule_model]
        rule_info = self.concise_Rule(train_list, rule_dic_list, -1.645)
        self.concised_rule, self.df_concise_rule = self.concised_rule_extraction(rule_dic_list, rule_info)
        for key, value in self.df_concise_rule.items():
            self.df_concise_rule = self.sort_homo(value)        
            
        rules = self.df_concise_rule
        subset = X.copy()
        covered_ = y.copy()
        covered_.loc[:] = None
        for col in rules.columns:
            rule = rules[col].loc['Rule'].split()
            pred_y = rules[col].loc['pred_y']
            var = rule[0::3]
            sign = rule[1::3]
            value = list(map(lambda v: v.strip(','),rule[2::3]))
            for num in range(len(var)):
                try: 
                    subset = subset.loc[self.comp(var[num], sign[num], value[num], subset),:]
                except (KeyError,IndexError):
                     continue
            covered_.loc[subset.index] = pred_y
            subset = X[covered_.isnull()].copy() 
            
        residual_idx = covered_[covered_.isnull()].index
        self.others_df = df_org.loc[residual_idx,:]            
    ############################################################################################
    def export_text(self):
        for key, values in self.concised_rule[0].items():
            print('IF', values[:-1], 'THEN', self.target_name,'=',self.CLASS_DICT[values[-1][0]], 
                  '  (homogeneity =',values[-1][1],', coverage =',values[-1][2],')')
    ############################################################################################
    def predict(self, dataset):         
        dataset=dataset.copy()
        colnm = dataset.columns
        X = dataset.loc[:,colnm [colnm != self.target_name]]
        y = dataset.loc[:, self.target_name]
        y.loc[:] = self.multi_to_integer(y)
        y = y.astype('int64')
        
        in_feature = list(dataset.columns[dataset.columns != self.target_name])
        cate_col = [col for col in in_feature \
                    if not np.issubdtype(X[col].dtype, np.number)]
        X_d = pd.get_dummies(dataset.loc[:,in_feature], columns=cate_col)    
        
        predict_class = y.copy()
        predict_class.loc[:] = None
        rules = self.df_concise_rule
        subset = X.copy()

        for col in rules.columns:
            rule = rules[col].loc['Rule'].split()
            pred_y = rules[col].loc['pred_y']
            var = rule[0::3]
            sign = rule[1::3]
            value = list(map(lambda v: v.strip(','),rule[2::3]))
            for num in range(len(var)):
                try: 
                    subset = subset.loc[self.comp(var[num], sign[num], value[num], subset),:]
                except (KeyError,IndexError):
                     continue
            predict_class.loc[subset.index] = pred_y
            subset = X[predict_class.isnull()].copy() 
            
        residual_predict_class = predict_class[predict_class.isnull()] 
        residual_idx = predict_class[predict_class.isnull()].index
        
        if len(residual_idx) > 0:
            predict_class.loc[residual_idx] = self.rf.predict(X_d.loc[residual_idx,:])
            predict_class = predict_class.astype('int64')

        accuracy = np.round(metrics.accuracy_score(y, predict_class),4)
        f1 = np.round(metrics.f1_score(y, predict_class, average='macro'),4)

        predict_class = predict_class.replace(self.CLASS_DICT)
        return predict_class, accuracy, f1