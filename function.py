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
    pat = ['dep','anx','sts']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax[i].hist(data['y_'+pat[i]],range=(0,4.5),bins=9,\
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
    #corr_matrix = data[index_q()].corr()
    #sns.set(rc={'figure.figsize':(30,30)})
    #sns.heatmap(data=corr_matrix,annot=True)

def find_best_cut(y_test_predict,Y_test):
    patology = ['Depression','Anxiety','Stress']
    best_cut = []
    print('The best cuts and the performances')
    for C in range(3):
        cut = 0.01
        perf = []
        for m in range(90):
            k = 0 
            y_lin_reg = []
            for i in range(len(y_test_predict[:,C])):
                if(y_test_predict[i,C]-int(y_test_predict[i,C]) < cut):
                    y_lin_reg.append(int(y_test_predict[i,C]))
                else:
                    y_lin_reg.append(int(y_test_predict[i,C]+1))
                if(y_lin_reg[i] == Y_test.iat[i,C]):
                    k += 1

            perf.append(k/len(y_test_predict[:,C]))
            cut = cut + 0.01
        # best cuts e performances
        print('- '+patology[C])
        print(((perf.index(max(perf))+1)/10),max(perf))
        best_cut.append((perf.index(max(perf))+1)/10)
        
    return best_cut

    
def linear_regression(data):
    # usefull tools
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # creating the X and Y dataframes
    X = data[index_q()]
    Y = data[['y_dep','y_anx','y_sts']]
    
    # splits the training and test data set in 80% : 20%
    # assign random_state to any value.This ensures consistency.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
    print('X, Y train/test shape')
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print("\n")
    
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    # model evaluation for training set

    y_train_predict = lin_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

    # model evaluation for testing set

    y_test_predict = lin_model.predict(X_test)
    # root mean square error of the model
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

    # r-squared score of the model
    r2 = r2_score(Y_test, y_test_predict)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")
    
    best_cut = find_best_cut(y_test_predict,Y_test)
    
    #print(best_cut)
    
    # using the best cuts
    lin_reg = pd.DataFrame()
    pat_index = ['y_dep','y_anx','y_sts']
    m = 0
    for m in range(3):
        y_lin_reg = []
        for i in range(len(y_test_predict[:,m])):
            if(y_test_predict[i,m]-int(y_test_predict[i,m]) < best_cut[m]):
                y_lin_reg.append(int(y_test_predict[i,m]))
            else:
                y_lin_reg.append(int(y_test_predict[i,m]+1))

        if m == 0:
            lin_reg.insert(m,pat_index[m],y_lin_reg)
        if m == 1:
            lin_reg.insert(m,pat_index[m],y_lin_reg)
        if m == 2:
            lin_reg.insert(m,pat_index[m],y_lin_reg)
            
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    #plotting results
    patology = ['Depression','Anxiety','Stress']
    pat_index = ['y_dep','y_anx','y_sts']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax2[i].hist(lin_reg[pat_index[i]],range=(0,4.5),bins=9,\
                        histtype='step',orientation='horizontal',align='left',color='grey',label='Y_pred')
        ax2[i].hist(Y_test[pat_index[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',color=col[i],label='Y_test')
        ax2[i].set_xlabel('# of cases')
        ax2[i].set_ylabel(patology[i]+' levels')
        ax2[i].set_title(patology[i]+' hist')
        ax2[i].text(100, -0.05, 'Normal', c='r')
        ax2[i].text(100, 0.95, 'Mild', c='r')
        ax2[i].text(100, 1.95, 'Moderate', c='r')
        ax2[i].text(100, 2.95, 'Severe', c='r')
        ax2[i].text(100, 3.95, 'Extremely Severe', c='r')
        ax2[i].legend()

    plt.show()