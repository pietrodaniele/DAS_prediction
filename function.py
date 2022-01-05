import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def reading_file(FILENAME):
    data = pd.read_csv(FILENAME)
    
    return data

def index_q():
    index_question = []
    for i in range(42):
        index_question.append('Q'+str(i+1)+'A')
            
    return index_question

def index_usefull():
    index = index_q()
    index_score = ['depression_score','anxiety_score','stress_score']
    index.append(index_score)
    
    return index

def index_train():
    index = index_q()
    index_y = ['y_dep','y_anx','y_sts']
    index.append(index_score)
    
    return index

def y_calc(score,y_col,limit):

    if score <= limit[0]:
        y_col.append(0)
    if score >= limit[1] and score <= limit[2]:
        y_col.append(1)
    if score >= limit[3] and score <= limit[4]:
        y_col.append(2)
    if score >= limit[5] and score <= limit[6]:
        y_col.append(3)
    if score >= limit[7]:
        y_col.append(4)
    

def analysis_df():
    # reading the dataframe
    data = reading_file('Data/data.csv')
    # adding score using the table [see ref in the introduction]
    # depression, anxiety and stress index and score limits
    d_index = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
    d_limit = [9, 10, 13, 14, 20, 21, 27, 28]
    a_index = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
    a_limit = [7, 8, 9, 10, 14, 15, 19, 20]
    s_index = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
    s_limit = [14, 15, 18, 19, 25, 26, 33, 34]

    # depression, anxiety and stress colums
    d_col = []; y_dep = []
    a_col = []; y_anx = []
    s_col = []; y_sts = []

    for i in range(len(data)):
        dep = 0; anx = 0; sts = 0;
        # depressione score
        for j in d_index:
            dep += (data['Q'+str(j)+'A'][i]-1)
        # anxiety score
        for k in a_index:
            anx += (data['Q'+str(k)+'A'][i]-1)
        # stress score
        for l in s_index:
            sts += (data['Q'+str(l)+'A'][i]-1)

        # adding scores and y
        ## depression
        d_col.append(dep)
        y_calc(dep,y_dep,d_limit)
        ## anxiety
        a_col.append(anx)
        y_calc(anx,y_anx,a_limit)
        ## stress
        s_col.append(sts)
        y_calc(sts,y_sts,s_limit)
    # adding score colums
    data.insert(len(data.keys()),'depression_score',d_col)
    data.insert(len(data.keys()),'anxiety_score',a_col)
    data.insert(len(data.keys()),'stress_score',s_col)
    data.insert(len(data.keys()),'y_dep',y_dep)
    data.insert(len(data.keys()),'y_anx',y_anx)
    data.insert(len(data.keys()),'y_sts',y_sts)
    
    data
    
    return data

def show_data(data):
    # showing the dsa data
    # creating a subplot
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    patology = ['Depression','Anxiety','Stress']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax[i].hist(data['y_dep'],range=(0,4.5),bins=9,\
                        histtype='step',orientation='horizontal',align='left',color=col[i])
        ax[i].set_xlabel('# of cases')
        ax[i].set_ylabel(patology[i]+' levels')
        ax[i].set_title(patology[i]+' hist')
        ax[i].text(1000, -0.05, 'Normal', c='r')
        ax[i].text(1000, 0.95, 'Mild', c='r')
        ax[i].text(1000, 1.95, 'Moderate', c='r')
        ax[i].text(1000, 2.95, 'Severe', c='r')
        ax[i].text(1000, 3.95, 'Extremely Severe', c='r')

    plt.show()
    
    # plotting correlation matrix
    corr_matrix = data[index_q()].corr()
    sns.set(rc={'figure.figsize':(30,30)})
    sns.heatmap(data=corr_matrix,annot=True)