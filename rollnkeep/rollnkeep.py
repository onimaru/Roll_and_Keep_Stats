from numpy import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats


# ============================================================= #
def dice(sides,rolls):
    roll = lambda sides: random.randint(1,sides+1)
    n_rolls = [roll(sides) for i in range(rolls)]
    return n_rolls

    '''
    Example:
    >>> dice(10,10)
    >>> [10, 1, 3, 7, 9, 4, 6, 3, 4, 1]    
    '''
    
# ============================================================= #    
def roll_n_keep(sides,rolls,keep):
    result = dice(sides,rolls)[:keep]
    result_sum = sum(result)
    return result

    '''
    Example:
    >>> roll_n_keep(10,5,3)
    >>> [4, 1, 5]
    '''

# ============================================================= #    
def final_roll(sides,rolls,keep,explode,explode_value=10):
    if explode == False:
        x = roll_n_keep(sides,rolls,keep)
    while explode:
        x = roll_n_keep(sides,rolls,keep)
        n_x = x.count(explode_value)
        while n_x:
            ex_x= roll_n_keep(sides,min(n_x,keep),min(n_x,keep))
            n_x = ex_x.count(explode_value)
            x = x + ex_x
            if n_x == 0:
                explode=False
        if n_x == 0:
                explode=False
    return sum(x)

    '''
    Example:
    >>> final_roll(10,5,3,True)
    >>> 14
    '''
    
# ============================================================= #
def roll_dist(sides,rolls,keep,explode,explode_value,n_tries):
    data = []
    for i in range(n_tries):
        data += [final_roll(sides,rolls,keep,explode)]
    return data

    '''
    Example:
    >>> data = roll_dist(10,5,3,True,10,10)
    >>> data
    >>> [23, 13, 26, 16, 15, 32, 14, 17, 19, 33]
    '''
    
# ============================================================= #
def plot_roll(sides,rolls,keep,n_tries):
    data = roll_dist(sides,rolls,keep,True,10,n_tries)
    mean = np.round(np.mean(data),1)
    median = np.round(np.median(data),1)
    moda = stats.mode(data)[0][0]
    line_max = data.count(moda)/n_tries
    plt.figure(figsize=(22,8))
    bins = int(np.max(data)-np.min(data))
    sns.distplot(data,bins=bins);
    plt.title('Distribution: {}k{}'.format(rolls,keep),fontsize=18)
    plt.grid(True,which='both')
    plt.plot([mean,mean],[0,line_max],'r--',label='Mean: {}'.format(mean))
    plt.plot([median,median],[0,line_max],'b--',label='Median: {}'.format(median))
    plt.plot([moda,moda],[0,line_max],'k--',label='Mode: {}'.format(moda))
    plt.legend(loc=0)
    plt.xticks(range(0,max(data)),rotation=90,fontsize=15)
    plt.xlabel('Roll values',fontsize=12)
    plt.xlim(0,100)
    plt.show()
    
    '''
    Example:
    >>> plot_roll(10,5,3,100000)
    >>> 'Image will show'
    '''

# ============================================================= #
def percentages_print(data):
    bounds = range(5,80,5)
    print('TN - Probability')
    for bound in bounds:
        prob = 1 - sum(i <= bound for i in data)/len(data)
        print('{} - {:.2%}'.format(bound,prob))

    '''
    Example:
    >>> percentages_print(data)
    >>> TN - Probability
         5 - 98.98%
        10 - 88.10%
        15 - 60.49%
        20 - 31.95%
        25 - 15.68%
        30 - 7.09%
        35 - 3.04%
        40 - 1.31%
        45 - 0.48%
        50 - 0.18%
        55 - 0.08%
        60 - 0.03%
        65 - 0.02%
        70 - 0.00%
        75 - 0.00%
    '''
    
# ============================================================= #
def percentages_dict(data):
    bounds = range(5,90,5)
    data_dict = {}
    for bound in bounds:
        prob = np.round((1 - sum(i <= bound for i in data)/len(data))*100,2)
        data_dict.update({bound:prob})
    return data_dict

    '''
    Example:
    >>> percentages_dict(roll_dist(10,5,3,True,10,100000))
    >>> {5: 98.97,
         10: 87.92,
         15: 60.28,
         20: 31.87,
         25: 15.47,
         30: 7.0,
         35: 3.09,
         40: 1.3,
         45: 0.5,
         50: 0.19,
         55: 0.08,
         60: 0.04,
         65: 0.01,
         70: 0.0,
         75: 0.0,
         80: 0.0,
         85: 0.0}
    '''

# ============================================================= #
def percentages(data):
    bounds = range(5,90,5)
    p = []
    for bound in bounds:
        prob = np.round((1 - sum(i <= bound for i in data)/len(data))*100,2)
        p.append(prob)
    return [p]

    '''
    Example:
    >>> percentages(roll_dist(10,5,3,True,10,100000))
    >>> [[99.03,
          88.03,
          60.22,
          31.94,
          15.6,
          6.95,
          2.88,
          1.22,
          0.46,
          0.19,
          0.08,
          0.02,
          0.01,
          0.0,
          0.0,
          0.0,
          0.0]]
    '''

# ============================================================= #
def set_dataframe(lower_tn_bound,upper_tn_bound,step):
    TN = [i for i in np.arange(lower_tn_bound,upper_tn_bound,step)]
    df_xky = pd.DataFrame(columns=['TN to be hit'],data = TN)
    return df_xky
    
    '''
    Example:
    >>> set_dataframe(5,90,5)
    >>> 	TN to be hit
        0	5
        1	10
        2	15
        3	20
        4	25
        5	30
        6	35
        7	40
        8	45
        9	50
        10	55
        11	60
        12	65
        13	70
        14	75
        15	80
        16	85
    '''

# ============================================================= #
def make_roll_df(save_df):
    for r in range(1,11):
        for k in range(1,11):
            if k > r: break
            df_xky['{}k{}'.format(r,k)] = np.array(percentages(roll_dist(10,r,k,True,10,100000))).T
    if save_df:
        df_xky.to_csv('roll_perc_table_xky.csv',index=False)
    return df_xky
    

    '''
    Example:
    >>> make_roll_df(True) - will create a csv file will all roll percentages for the
    TN to be hit range.
    '''

# ============================================================= #    
def save_df2html(dataframe,filename):
    dataframe.to_html('{}.html'.format(filename))
    print('Done!')

# ============================================================= #   