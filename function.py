import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# usefull tools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
    

def reading_file(FILENAME):
    data = pd.read_csv(FILENAME)
    
    return data

def index_q():
    index_question = []
    for i in range(42):
        index_question.append('Q'+str(i+1)+'A')
            
    return index_question

def index_reduction(data,corr_limit):
    # correlation matrix
    corr = data[index_q()].corr()
    
    index = index_q()
    index_red = []
    red = []

    for i in range(len(index)):
        for j in range(len(index)):
            if i !=j and corr.iat[i,j] >= corr_limit:
                red.append(f'Q{j+1}A')

    for i in range(len(index)):
        k = 0
        for j in range(len(red)):
            if index[i] == red[j]:
                k = 1
        if k == 0:
            index_red.append(index[i])
    return index_red

def train_test_splitting(data,percent):
    # lib
    from sklearn.model_selection import train_test_split
    
    # creating the X and Y dataframes
    X = data[index_q()]
    Y = data[['y_dep','y_anx','y_sts']]
    
    # splits the training and test data set in 80% : 20%
    # assign random_state to any value.This ensures consistency.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = percent, random_state=5)
    print('X, Y train/test shape')
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print("\n")
    
    return X_train, X_test, Y_train, Y_test

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
    corr_matrix = data[index_q()].corr()
    corr_matrix.style.background_gradient(cmap='coolwarm')
    
    return

def find_best_cut(y_test_predict,Y_test):
    patology = ['Depression','Anxiety','Stress']
    best_cut = []
    # plotting perf(cut)
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    patology = ['Depression','Anxiety','Stress']
    pat_index = ['y_dep','y_anx','y_sts']
    col = ['b','g','purple']
    
    print('The best cuts and the performances')
    for C in range(3):
        cut = 0.01
        perf = []
        cut_array = []
        for m in range(90):
            k = 0 
            y_lin_reg = []
            for i in range(len(y_test_predict[:,C])):
                if(y_test_predict[i,C]-int(y_test_predict[i,C]) < cut):
                    y_lin_reg.append(int(y_test_predict[i,C]))
                else:
                    y_lin_reg.append(int(y_test_predict[i,C]+1))
                #if(y_lin_reg[i] == Y_test.iat[i,C]):
                #    k += 1
            
            cut_array.append(cut)
            perf.append(accuracy_score(y_lin_reg,Y_test[pat_index[C]]))
            cut = cut + 0.01
            
        ax2[C].plot(cut_array,perf,color=col[C],label='Perf')
        ax2[C].set_xlabel('Cut')
        ax2[C].set_ylabel('Performance')
        ax2[C].set_title(patology[C]+' classfication perf.')
        ax2[C].legend()
        
        # best cuts e performances
        print('- '+patology[C])
        print(((perf.index(max(perf))+1)/100),max(perf))
        best_cut.append((perf.index(max(perf))+1)/100)
    
    acc = sum(perf)/len(perf)
    print("\n")
    print(f"The model accuracy = {acc}")
    plt.show()
    return best_cut

def R_to_class(prediction, best_cut):
    lin_reg = pd.DataFrame()
    pat_index = ['y_dep','y_anx','y_sts']
    m = 0
    for m in range(3):
        y_lin_reg = []
        for i in range(len(prediction[:,m])):
            if(prediction[i,m]-int(prediction[i,m]) < best_cut[m]):
                y_lin_reg.append(int(prediction[i,m]))
            else:
                y_lin_reg.append(int(prediction[i,m]+1))

        if m == 0:
            lin_reg.insert(m,pat_index[m],y_lin_reg)
        if m == 1:
            lin_reg.insert(m,pat_index[m],y_lin_reg)
        if m == 2:
            lin_reg.insert(m,pat_index[m],y_lin_reg)
    
    return lin_reg
    
def linear_regression(data, X_train, X_test, Y_train, Y_test):
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    # model evaluation for training set
    print("* Training")
    print("\n")
    y_train_predict = lin_model.predict(X_train)
    best_cut_train = find_best_cut(y_train_predict,Y_train)
    y_train_class = R_to_class(y_train_predict,best_cut_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_class)))
    r2 = r2_score(Y_train, y_train_class)

    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

    # model evaluation for testing set
    print("* Test")
    print("\n")
    y_test_predict = lin_model.predict(X_test)
    best_cut = find_best_cut(y_test_predict,Y_test)
    y_test_class = R_to_class(y_test_predict,best_cut)
    # root mean square error of the model
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_class)))

    # r-squared score of the model
    r2 = r2_score(Y_test, y_test_class)
    
    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")
            
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    #plotting results
    patology = ['Depression','Anxiety','Stress']
    pat_index = ['y_dep','y_anx','y_sts']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax2[i].hist(y_test_class[pat_index[i]],range=(0,4.5),bins=9,\
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
    
    return y_test_class

def GNB_classification(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()

    y_pat = ['y_dep','y_anx','y_sts']
    acc = []
    Y_pred = pd.DataFrame()
    for i in range(3):
        print('* '+y_pat[i])
        print('\n')
        
        gnb.fit(X_train, np.ravel(Y_train[[y_pat[i]]])) # using ravel in order to pass a 1D array (not a 1D colum)
        
        # model evaluation for training set
        y_train_predict = gnb.predict(X_train)
        rmse = (np.sqrt(mean_squared_error(Y_train[[y_pat[i]]], y_train_predict)))
        r2 = r2_score(Y_train[[y_pat[i]]], y_train_predict)

        print("The model performance for training set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        # model evaluation for testing set

        y_test_predict = gnb.predict(X_test)
        # root mean square error of the model
        rmse = (np.sqrt(mean_squared_error(Y_test[[y_pat[i]]], y_test_predict)))

        # r-squared score of the model
        r2 = r2_score(Y_test[[y_pat[i]]], y_test_predict)

        print("The model performance for testing set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        Y_pred.insert(i,y_pat[i],y_test_predict)
        acc.append(accuracy_score(Y_test[[y_pat[i]]],Y_pred[[y_pat[i]]]))
        print(acc[i])
        print('######################################')
        print("\n")
        
    
    print(f"The model accuracy = {sum(acc)/len(acc)}")
    
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    #plotting results
    patology = ['Depression','Anxiety','Stress']
    pat_index = ['y_dep','y_anx','y_sts']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax2[i].hist(Y_pred[pat_index[i]],range=(0,4.5),bins=9,\
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
    
    return Y_pred

def KNN_classification(X_train, X_test, Y_train, Y_test):
    clf = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')

    y_pat = ['y_dep','y_anx','y_sts']
    acc = []
    Y_pred = pd.DataFrame()
    for i in range(3):
        print('* '+y_pat[i])
        print('\n')
        
        clf.fit(X_train, np.ravel(Y_train[[y_pat[i]]])) # using ravel in order to pass a 1D array (not a 1D colum)
        
        # model evaluation for training set
        y_train_predict = clf.predict(X_train)
        rmse = (np.sqrt(mean_squared_error(Y_train[[y_pat[i]]], y_train_predict)))
        r2 = r2_score(Y_train[[y_pat[i]]], y_train_predict)

        print("The model performance for training set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        # model evaluation for testing set

        y_test_predict = clf.predict(X_test)
        # root mean square error of the model
        rmse = (np.sqrt(mean_squared_error(Y_test[[y_pat[i]]], y_test_predict)))

        # r-squared score of the model
        r2 = r2_score(Y_test[[y_pat[i]]], y_test_predict)

        print("The model performance for testing set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        Y_pred.insert(i,y_pat[i],y_test_predict)
        acc.append(accuracy_score(Y_test[[y_pat[i]]],Y_pred[[y_pat[i]]]))
        print(acc[i])
        print('######################################')
        print("\n")
        
    
    print(f"The model accuracy = {sum(acc)/len(acc)}")
    
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    #plotting results
    patology = ['Depression','Anxiety','Stress']
    pat_index = ['y_dep','y_anx','y_sts']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax2[i].hist(Y_pred[pat_index[i]],range=(0,4.5),bins=9,\
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
    
    return Y_pred

def classification(model,X_train, X_test, Y_train, Y_test, index_T):
    # variables
    y_pat = ['y_dep','y_anx','y_sts']
    acc = []
    Y_pred = pd.DataFrame()
    for i in range(3):
        print('* '+y_pat[i])
        print('\n')
        
        model.fit(X_train[index_T[i]], np.ravel(Y_train[[y_pat[i]]])) # using ravel in order to pass a 1D array (not a 1D colum)
        
        # model evaluation for training set
        y_train_predict = model.predict(X_train[index_T[i]])
        rmse = (np.sqrt(mean_squared_error(Y_train[[y_pat[i]]], y_train_predict)))
        r2 = r2_score(Y_train[[y_pat[i]]], y_train_predict)

        print("The model performance for training set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        # model evaluation for testing set

        y_test_predict = model.predict(X_test[index_T[i]])
        # root mean square error of the model
        rmse = (np.sqrt(mean_squared_error(Y_test[[y_pat[i]]], y_test_predict)))

        # r-squared score of the model
        r2 = r2_score(Y_test[[y_pat[i]]], y_test_predict)

        print("The model performance for testing set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        Y_pred.insert(i,y_pat[i],y_test_predict)
        acc.append(accuracy_score(Y_test[[y_pat[i]]],Y_pred[[y_pat[i]]]))
        print(acc[i])
        print('######################################')
        print("\n")
        
    
    print(f"The model accuracy = {sum(acc)/len(acc)}")
    
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    
    #plotting results
    patology = ['Depression','Anxiety','Stress']
    pat_index = ['y_dep','y_anx','y_sts']
    col = ['b','g','purple']

    # plotting in each subplot
    for i in range(3):
        ax2[i].hist(Y_pred[pat_index[i]],range=(0,4.5),bins=9,\
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
    
    return Y_pred

def error_show(model,data,X_train, X_test, Y_train, Y_test):
    pat_index = ['y_dep','y_anx','y_sts']
    fig, ax2 = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
    for k in range(3):
        # error
        error_train = []
        error_test = []
        x = []
        cut = 0.4
        for i in range(30):
            index_2 = index_reduction(data,cut)
            x.append(len(index_2))
            # fit
            model.fit(X_train[index_2], np.ravel(Y_train[pat_index[k]]))

            # training
            y_train_predict = model.predict(X_train[index_2])
            error_train.append(mean_squared_error(Y_train[pat_index[k]], y_train_predict))
            # test
            y_test_predict = model.predict(X_test[index_2])
            error_test.append(mean_squared_error(Y_test[pat_index[k]], y_test_predict))

            # cut 
            cut = cut + 0.02
    
        ax2[k].plot(x,error_train,label='train')
        ax2[k].plot(x,error_test,label='test')
        ax2[k].set_title(f'Training and test errors => '+pat_index[k])  
        ax2[k].set_xlabel('# of index')
        ax2[k].set_ylabel('Errors')
        ax2[k].legend(loc='best')
        
    plt.show()
    
def show_results(Y_test, Y_lin, Y_gbn, Y_log, Y_knn, Y_dt, Y_svm):
    y_pat = ['y_dep','y_anx','y_sts']
    patology = ['Depression','Anxiety','Stress']
    for i in range(3):
        fig = plt.figure(figsize=(10,8))
        #histos
        plt.hist(Y_test[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',label='Y_true')
        plt.hist(Y_lin[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',alpha=0.5,label='Y_lin')
        plt.hist(Y_gbn[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',alpha=0.5,label='Y_gbn')
        plt.hist(Y_log[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',alpha=0.5,label='Y_log')
        plt.hist(Y_knn[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',alpha=0.5,label='Y_knn')
        plt.hist(Y_dt[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',alpha=0.5,label='Y_dt')
        plt.hist(Y_svm[y_pat[i]],range=(0,4.5),bins=9,\
                            histtype='step',orientation='horizontal',align='left',alpha=0.5,label='Y_svm')

        plt.title(y_pat[i]+' results')
        plt.xlabel('# of cases')
        plt.ylabel(patology[i]+' levels')
        plt.text(100, -0.05, 'Normal', c='r')
        plt.text(100, 0.95, 'Mild', c='r')
        plt.text(100, 1.95, 'Moderate', c='r')
        plt.text(100, 2.95, 'Severe', c='r')
        plt.text(100, 3.95, 'Extremely Severe', c='r')
        plt.legend(loc='center right')
        plt.show()
       
    # accuracy depression
    accuracy_dep = [accuracy_score(Y_test['y_dep'],Y_lin['y_dep']),\
                    accuracy_score(Y_test['y_dep'],Y_gbn['y_dep']),\
                    accuracy_score(Y_test['y_dep'],Y_log['y_dep']),\
                    accuracy_score(Y_test['y_dep'],Y_knn['y_dep']),\
                    accuracy_score(Y_test['y_dep'],Y_dt['y_dep']),\
                    accuracy_score(Y_test['y_dep'],Y_svm['y_dep'])]
    # accuracy anxiety
    accuracy_anx = [accuracy_score(Y_test['y_anx'],Y_lin['y_anx']),\
                    accuracy_score(Y_test['y_anx'],Y_gbn['y_anx']),\
                    accuracy_score(Y_test['y_anx'],Y_log['y_anx']),\
                    accuracy_score(Y_test['y_anx'],Y_knn['y_anx']),\
                    accuracy_score(Y_test['y_anx'],Y_dt['y_anx']),\
                    accuracy_score(Y_test['y_anx'],Y_svm['y_anx'])]
    # accuracy stress
    accuracy_sts = [accuracy_score(Y_test['y_sts'],Y_lin['y_sts']),\
                    accuracy_score(Y_test['y_sts'],Y_gbn['y_sts']),\
                    accuracy_score(Y_test['y_sts'],Y_log['y_sts']),\
                    accuracy_score(Y_test['y_sts'],Y_knn['y_sts']),\
                    accuracy_score(Y_test['y_sts'],Y_dt['y_sts']),\
                    accuracy_score(Y_test['y_sts'],Y_svm['y_sts'])]
    # tot accuracy
    accuracy_tot = []
    for j in range(len(accuracy_dep)):
        accuracy_tot.append((accuracy_dep[j]+accuracy_anx[j]+accuracy_sts[j])/3)
    # results/accuracy dataframe
    results = pd.DataFrame()
    results.insert(len(results.keys()),'dep_acc',accuracy_dep)
    results.insert(len(results.keys()),'anx_acc',accuracy_anx)
    results.insert(len(results.keys()),'sts_acc',accuracy_sts)
    results.insert(len(results.keys()),'tot_acc',accuracy_tot)

    results.rename({0:'lin', 1:'GBN', 2:'log', 3:'KNN', 4:'DT', 5:'SVM'}, axis=0, inplace=True)
    
    return results