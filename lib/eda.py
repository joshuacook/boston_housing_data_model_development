import seaborn as sns
import matplotlib.pyplot as plt

def feature_info(dataframe, col, display_unique=True):
    if not display_unique:
        unique = ''
    else:
        unique = dataframe[col].unique()
    print("{:20} {:10} null values: {} {}"
          .format(col,
                  str(dataframe[col].dtype), 
                  sum(dataframe[col].isnull()), 
                  str(unique)))

def feature_null_info(dataframe, col, dataset, display_unique):
    if not display_unique:
        unique = ''
    else:
        unique = dataframe[col].unique()
    print("{:6} {:20} {:10} null values: {} {}"
          .format(dataset,
                  col,
                  str(dataframe[col].dtype), 
                  sum(dataframe[col].isnull()), 
                  str(unique)))
    
def feature_null_info_train_test(dataframe_train, dataframe_test, col, display_unique=True):
    feature_null_info(dataframe_train, col, 'train', display_unique)
    feature_null_info(dataframe_test, col, 'test', display_unique)    
    
def identify_nulls(dataframe_train, dataframe_test):
    columns = list(dataframe_train.columns)
    columns.remove('SalePrice')
    for col in columns:
        null_train_values = sum(dataframe_train[col].isnull())
        datatype = str(dataframe_train[col].dtype)
        null_test_values = sum(dataframe_test[col].isnull())
        if null_train_values > 0: print("{:20} {:10} null values: {:6} null test values: {}"
                                  .format(col, datatype, null_train_values, null_test_values))    
            
def feat_dist_plot(dataframe_train, dataframe_test, column_name, display_test=True):
    col = 1 + int(display_test)
    fig = plt.figure(figsize=(6*col,3))
    fig.add_subplot(1,col,1)
    sns.distplot(dataframe_train[column_name].dropna().values)
    plt.axvline(dataframe_train[column_name].median(), c='red')
    plt.axvline(dataframe_train[column_name].mean(), c='green')
    if display_test:
        fig.add_subplot(1,col,2)
        sns.distplot(dataframe_test[column_name].dropna())
        plt.axvline(dataframe_test[column_name].median(), c='red')
        plt.axvline(dataframe_test[column_name].mean(), c='green')            