# -*- coding: utf-8 -*-
"""
@author: KP
"""
def statistical_test(groups, pairing = 'unpaired'): #function for statistical testing
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu, wilcoxon
    from pingouin import homoscedasticity, normality, sphericity, rm_anova, friedman, anova, welch_anova, kruskal, pairwise_tukey
    from scikit_posthocs import posthoc_tamhane, posthoc_nemenyi_friedman, posthoc_dunn
    
    if isinstance(groups, pd.core.frame.DataFrame): #if provided data is a data frame
        groups = groups.to_dict(orient='list') #make it a dictionary as the whole function works using this
        
    group_names = list(groups.keys()) #a list of group names
    
    number_of_groups = len(groups) #calculate the number of groups
    n = [] #a list for n for each group
    for group_name in groups: #group_name='Control'
        n_temp = 0 if np.array(groups[group_name]).shape[0] == 0 else np.array(groups[group_name]).shape[0] #calculate n for each group
        n.append(n_temp) #append it to the list
        
    if (pairing == 'paired') & (all(x == n[0] for x in n) == False): #checks if n in both groups the same if 'paired'
        pairing = 'unpaired' #if not then switch to unpaired testing
        print('Warning: the test is supposed to be paired but the group sizes are different - the unpaired versions of the tests will be performed instead.') #and give a warning to the user
    
    if (number_of_groups >= 2): #for at least two groups
        if (min(n) >= 3) & ~np.all(np.array(list(groups.values())) == 0): #check if n at least 3 in each group
            groups_long = pd.DataFrame(columns=['variable', 'value']) #create a data frame for a long format data
            for group_name in groups: #for each group
                df_temp = pd.DataFrame({'variable' : group_name, 'value' : np.array(groups[group_name], dtype=np.float64)}) #add it to a data frame
                groups_long = pd.concat([groups_long, df_temp]) #append
            groups_long.reset_index(inplace=True, drop=True) #reset indexes
                    
            shapiro = normality(data=groups_long, dv='value', group='variable', method='shapiro') #perform a shapiro normality test
            p_shapiro = shapiro['pval'].values #take the p values
            
            levene = homoscedasticity(data=groups_long.dropna(), dv='value', group='variable', method='levene') #perform levene homoscedasticity test
            p_levene = levene['pval'].values[0] #take the p value
                 
            if number_of_groups == 2: #if two groups
                group_one = pd.Series(groups[list(groups.keys())[0]], name=list(groups.keys())[0]) #take data for group one
                group_two = pd.Series(groups[list(groups.keys())[1]], name=list(groups.keys())[1]) #take data for group two
                n_one, n_two = group_one.shape[0], group_two.shape[0] #calculate N
                
                if min(p_shapiro) > 0.05: #if normal distribution for both groups
                    if pairing == 'paired': #if paired
                        stat, p_val = ttest_rel(group_one, group_two, alternative='two-sided') #paired t test
                        df = n_one - 1 #calculate DF
                        test_name = 't-test (paired)' #name of the test
                    elif pairing == 'unpaired': #if unpaired
                        if p_levene > 0.05: #if variances equal
                            stat, p_val = ttest_ind(group_one, group_two, alternative='two-sided', equal_var=True) #t test unpaired
                            df = n_one + n_two - 2 #calculate DF
                            test_name = 't-test (equal variances, unpaired)' #name of the test
                        elif p_levene <= 0.05: #if variances not equal
                            stat, p_val = ttest_ind(group_one, group_two, alternative='two-sided', equal_var=False) #Welch's test
                            var_one, var_two = np.var(group_one, ddof=1), np.var(group_two, ddof=1) #calculate variances of the groups
                            df = ((var_one / n_one + var_two / n_two) ** 2) / ((var_one ** 2 / (n_one ** 2 * (n_one - 1))) + (var_two ** 2 / (n_two ** 2 * (n_two - 1)))) #Welch-Satterthwaite equation for DF
                            test_name = 'Welch t-test (unequal variances, unpaired)' #name of the test
                else: #if data not normally distributed
                    if (pairing == 'paired') & ~np.all((group_one - group_two) == 0): #if paired
                        stat, p_val = wilcoxon(group_one, group_two, alternative='two-sided') #Wilcoxon test
                        df = np.nan #no DF 
                        test_name = 'Wilcoxon signed-rank test (paired)' #name of the test
                    else: #if not paired
                        if p_levene > 0.05: #if variances equal
                            stat, p_val = mannwhitneyu(group_one, group_two, alternative='two-sided', use_continuity=False) #Mann-Whitney test
                            test_name = 'Mann-Whitney U test (equal variances, unpaired)' #name of the test
                        elif p_levene <= 0.05: #if variances not equal
                            stat, p_val = mannwhitneyu(group_one, group_two, alternative='two-sided', use_continuity=True) #Mann-Whitney for data with differnet variances # stat, p_val, df
                            test_name = 'Mann-Whitney U test (unequal variances, unpaired)' #name of the test
                        df = np.nan #no DF
                        #df = (n_one * n_two) / (n_one + n_two)
            elif number_of_groups > 2: #if 3 or more groups
                post_hoc = np.nan #it will stay like this if no post hoc be required
                post_hoc_name = 'none' #it will stay like this if no post hoc be required
                
                if min(p_shapiro) > 0.05: #if normal distribution for both groups
                    if pairing == 'paired': #if paired
                        _, _, _, _, p_mauchly = sphericity(pd.DataFrame(groups), method='mauchly') #perform Mauchly's sphericity test
                        if p_mauchly > 0.05: #if the data is spherical
                            results = rm_anova(pd.DataFrame(groups), correction=False) #perform repeated measures ANOVA with no correction
                            p_val, df_one, df_two, stat = results['p-unc'].values[0], results['ddof1'].values[0], results['ddof2'].values[0], results['F'].values[0] #take the statistics
                            test_name = 'repeated-measures ANOVA (no correction)' #declare test name  
                        else: #if the data is not spherical
                            results = rm_anova(pd.DataFrame(groups), correction=True) #perform repeated measures ANOVA with Greenhouse-Geisser correction
                            p_val, df_one, df_two, stat = results['p-unc'].values[0], results['ddof1'].values[0], results['ddof2'].values[0], results['F'].values[0] #take the statistics
                            test_name = 'repeated-measures ANOVA (GG corrected)' #declare test name
                        if p_val <= 0.05: #if test significant perform a post hoc test
                            post_hoc_name = 'Tukey HSD test' #declare a post hoc test name    
                            post_hoc_temp = pairwise_tukey(data=groups_long, dv='value', between='variable') #it performs Tukey HSD when sample size equal (and they have to be here)    
                            post_hoc = pd.DataFrame(columns=group_names, index=group_names) #create a data frame when 'matrix' of p values will be stored
                            for comp in post_hoc_temp.iterrows(): #add p values to this matrix
                                post_hoc.loc[comp[1]['A'], comp[1]['B']] = comp[1]['p-tukey']
                                post_hoc.loc[comp[1]['B'], comp[1]['A']] = comp[1]['p-tukey']       
                    elif pairing == 'unpaired': #if unpaired
                        if p_levene > 0.05: #if variances equal
                            results = anova(data=groups_long, dv='value', between='variable') #perform ANOVA
                            p_val, df_one, df_two, stat = results['p-unc'].values[0], results['ddof1'].values[0], results['ddof2'].values[0], results['F'].values[0] #take the statistics
                            test_name = 'one-way ANOVA' #declare test name
                            if p_val <= 0.05: #if test significant perform a post hoc test
                                if all(x == n[0] for x in n): #if sample size equal
                                    post_hoc_name = 'Tukey HSD test' #declare a post hoc test name
                                else: #if sample size not equal
                                    post_hoc_name = 'Tukey-Kramer test' #declare a post hoc test name    
                                post_hoc_temp = pairwise_tukey(data=groups_long, dv='value', between='variable') #it performs Tukey HSD when sample size equal (and they have to be here)         
                                post_hoc = pd.DataFrame(columns=group_names, index=group_names) #create a data frame when 'matrix' of p values will be stored
                                for comp in post_hoc_temp.iterrows(): #add p values to this matrix
                                    post_hoc.loc[comp[1]['A'], comp[1]['B']] = comp[1]['p-tukey']
                                    post_hoc.loc[comp[1]['B'], comp[1]['A']] = comp[1]['p-tukey']      
                        elif p_levene <= 0.05: #if variances not equal
                            results = welch_anova(data=groups_long, dv='value', between='variable') #perform Welch ANOVA
                            p_val, df_one, df_two, stat = results['p-unc'].values[0], results['ddof1'].values[0], results['ddof2'].values[0], results['F'].values[0] #take the statistics
                            test_name = 'Welch ANOVA' #declare test name
                            if p_val <= 0.05: #if test significant perform a post hoc test
                                post_hoc_name = 'Tamhane\s T2 test' #declare a post hoc test name
                                post_hoc = posthoc_tamhane(groups_long, val_col='value', group_col='variable') #make a data frame from p values
                else: #if not notmally distributed
                    if pairing == 'paired': #if paired
                        results = friedman(pd.DataFrame(groups)) #perform Friedman's test
                        p_val, df_one, df_two, stat = results['p-unc'].values[0], results['ddof1'].values[0], np.nan, results['Q'].values[0] #take the statistics
                        test_name = 'Friedman test' #declare test name
                        if p_val <= 0.05: #if test significant perform a post hoc test
                            post_hoc_name = 'Nemenyi test' #declare a post hoc test name
                            post_hoc = posthoc_nemenyi_friedman(pd.DataFrame(groups)) #make a data frame from p values
                    elif pairing == 'unpaired': #if unpaired
                        #exec('stat, p_val = kruskal(' + groups_string + ')')
                        results = kruskal(data=groups_long, dv='value', between='variable') #perform Kruskal-Wallis test
                        p_val, df_one, df_two, stat = results['p-unc'].values[0], results['ddof1'].values[0], np.nan, results['H'].values[0] #take the statistics
                        test_name = 'Kruskal-Wallis H test' #declare test name
                        if p_val <= 0.05: #if test significant perform a post hoc test
                            post_hoc_name = 'Dunn\'s test' #declare a post hoc test name
                            post_hoc = posthoc_dunn(groups_long, val_col='value', group_col='variable') #make a data frame from p values
        else:
            test_name = 'none'
            post_hoc_name = 'none'
            stat = np.nan
            p_val = np.nan
            df = np.nan
            df_one = np.nan
            df_two = np.nan
            
        #calculate all basic statistics and add everything to a data frame
        statistics_summary = pd.DataFrame()
        for group_name in groups:
            statistics_summary[group_name + '_mean'] = [np.mean(groups[group_name], axis=0)]
        for group_name in groups:
            statistics_summary[group_name + '_SD'] = [np.std(groups[group_name], axis=0)]
        for group_name in groups:
            statistics_summary[group_name + '_SEM'] = [np.std(groups[group_name], axis=0) / np.sqrt(len(groups[group_name]))]
        for group_name in groups:
            statistics_summary[group_name + '_median'] = [np.median(groups[group_name], axis=0)]
        for group_name in groups:
            statistics_summary[group_name + '_Q1'] = [np.quantile(groups[group_name], 0.25, axis=0) if len(groups[group_name]) != 0 else np.nan]
        for group_name in groups:
            statistics_summary[group_name + '_Q3'] = [np.quantile(groups[group_name], 0.75, axis=0) if len(groups[group_name]) != 0 else np.nan]
        for group_name in groups:
            statistics_summary[group_name + '_N'] = [len(groups[group_name])]
        
        #add test results to a data frame
        statistics_summary['test'] = test_name
        statistics_summary['statistic'] = stat
        statistics_summary['p-value'] = p_val
        if number_of_groups == 2:
            statistics_summary['df'] = df
        elif number_of_groups > 2:
            statistics_summary['df_1'] = df_one
            statistics_summary['df_2'] = df_two
        
            #add post hoc data to a data frame
            statistics_summary['post-hoc test'] = post_hoc_name
            for i in range(len(group_names)): #this loop adds relevant post hoc comparisons to a data frame with statistics
                for j in range(i+1, len(group_names)):
                    pair_name = group_names[i] + '_vs_' + group_names[j]
                    if post_hoc_name != 'none':
                        pair = post_hoc.iloc[i, j]
                        statistics_summary[pair_name] = pair
                    else:
                        statistics_summary[pair_name] = np.nan 
    else:
        print('Error: at least two groups have to be provided!')
    
    return statistics_summary

