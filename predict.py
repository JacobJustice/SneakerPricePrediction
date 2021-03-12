from pprint import pprint
import pandas as pd
import argparse
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle

def clean_sneaker_data_for_ml(df):
    # drop duplicates
    df = df.drop_duplicates()

    # clean average_sale_price
    df = df[df['average_sale_price'].notna()]
    df['average_sale_price'] = df['average_sale_price'].str[1:]
    df['average_sale_price'] = df['average_sale_price'].str.replace(',','')
    df['average_sale_price'] = df['average_sale_price'].astype(int)
    df = df[df['average_sale_price'] < 1000]

    # clean retail_price
    df = df[df['retail_price'].notna()]
    df['retail_price'] = df['retail_price'].str[1:]
    df['retail_price'] = df['retail_price'].str.replace(',','')
    df['retail_price'] = df['retail_price'].astype(int)

    df = df[df['pixel0_r'].notna()]

    # drop unused columns

    df = df.drop(['url'
            , 'name'
            , 'ticker'
            , 'image_path'
            , 'release_date'
            , 'number_of_sales'
            , 'price_premium'
            , 'style_code'
            , 'retail_price'
            , 'colorway'], axis=1)


    return df

# INTENDED TO BE USED ON TRAINING SET
# 
# normalizes every column except average_sale_price and returns an unlabeled dataframe
def normalize_pixels(training_df, min_max_scaler=None):
    x = training_df.drop('average_sale_price',axis=1).values
    if (min_max_scaler is None):
        min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)
    #print(x_scaled)
    scaled_training_df = pd.DataFrame(x_scaled)

    scaled_training_df['average_sale_price'] = training_df['average_sale_price']
    return scaled_training_df, min_max_scaler


def load_df(path_to_csv):
    return pd.read_csv(path_to_csv)


# merges on ticker
def merge_df(df, flat_df):
    out_df = pd.merge(df, flat_df, on='ticker', how='outer'),
    return out_df


def make_training_and_validation(shuffle_seed=1234):
    parser = argparse.ArgumentParser(description='Auto-crop a directory of images')
    parser.add_argument('-df','--dataframe', required=True, help='dataframe containing scraped shoe data')
    parser.add_argument('-fdf','--flat_dataframe', required=True, help='dataframe containing image data')

    args = parser.parse_args(sys.argv[1:])

    df = load_df(args.dataframe)
    flat_df = load_df(args.flat_dataframe)

    merged_df = merge_df(df, flat_df)[0]
    #merged_df = clean_sneaker_data_for_ml(merged_df)

    # make dummie columns
    name_dummies = merged_df['name'].str.get_dummies(" ")
    name_dummies = name_dummies[['(PS)','(GS)','(W)','(TD)','High','Mid','Low']]

    # drop columns that don't have more than 5% 1s in the column
    #print(relevant_columns := name_dummies.loc[:, name_dummies.eq(1).sum().gt(name_dummies.shape[0]*.05)])

    final = pd.concat([merged_df, name_dummies], axis=1)
    final = clean_sneaker_data_for_ml(final)
    final = shuffle(final, random_state=shuffle_seed)

    final_validation = final[0:int(len(final)*.1)]
    final_validation.to_csv('validation_aj1.csv', index=False)

    final_training = final[int(len(final)*.1):]
    final_training.to_csv('training_aj1.csv', index=False)

    print(final)
    final.to_csv('final_aj1.csv', index=False)
#uncomment if you want to make a new training and validation set
#make_training_and_validation()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVR

df = load_df('./training_aj1.csv')
# normalize training set
df, min_max_scaler = normalize_pixels(df)
df_x = df.drop('average_sale_price', axis=1)
df_y = df['average_sale_price']

# create test set
test = df[0:int(len(df)*.2)]
training = df[int(len(df)*.2):]

training_x = training.drop('average_sale_price', axis=1)
training_y = training['average_sale_price']

test_x = test.drop('average_sale_price', axis=1)
test_y = test['average_sale_price']

#pca = PCA(n_components='mle')
#pca_df_x = pca.fit_transform(df_x, df_y)
#print("n_components", pca.n_components_)
#print("COVARIANCE", pca.get_covariance())
#
#svr = SVR()
#param_dist = {
    #'kernel':['linear','poly'],
    #'gamma':['scale','auto'],
    #'degree':[1,2,3,4,5,6,7],
    #'C':[.0001,.001,.01,.1,1,10,100,1000],
    #'epsilon':[.0001,.001,.01,.1,1,10,100,1000]
#}

#gscv = GridSearchCV(estimator=svr, param_grid=param_dist, cv=7,
#                    verbose=3)
#gscv.fit(df_x, df_y)

svr = SVR(C=1, degree=5, epsilon=10, gamma='scale', kernel='poly')

pprint(scores := cross_val_score(svr, df_x, df_y, cv=7))
print(np.array(scores).mean())
#print(gscv.best_params_)
