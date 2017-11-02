from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import argmax, array, logspace, nan
from pandas import concat, DataFrame

def split_fit_score(model, X, y, n=30):
    train_scores = []
    test_scores = []
    
    for i in tqdm(range(n)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        
    mean_train_score = array(train_scores).mean()
    mean_test_score = array(test_scores).mean()
    return model, mean_train_score, mean_test_score

def simple_alpha_grid_search(model, X, y, results_df, n=10):
    train_scores = []
    test_scores = []
    model_steps = list(model.named_steps.keys())
    if model_steps[-1] == 'linearregression':
        _, train_score, test_score = split_fit_score(model, X, y, n=n)
        best_alpha = nan
    else:
        logspace_ary = logspace(-5,5,11)
        for alpha in logspace_ary:
            model.steps[-1][1].alpha = alpha
            _, train_score, test_score = split_fit_score(model, X, y, n=n)
            train_scores.append(train_score)
            test_scores.append(test_score)

        best_alpha = logspace_ary[argmax(test_scores)]
        model.steps[-1][1].alpha = best_alpha
        model, train_score, test_score = split_fit_score(model, X, y, n=n)
    
    model_name = model_steps.pop()
    results_df = concat([results_df, 
                         DataFrame([{'model': model,
                                     'model_name': model_name, 
                                     'data_preprocessing': model_steps, 
                                     'alpha': best_alpha, 
                                     'train_score':train_score, 
                                     'test_score':test_score}])])
    
    return results_df, train_scores, test_scores