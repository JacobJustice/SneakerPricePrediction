import pandas as pd
import argparse
import sys

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

    df['profit'] = df['average_sale_price'] - df['retail_price']

    return df

def load_df(path_to_csv):
    return pd.read_csv(path_to_csv)

# merges on ticker
def merge_df(df, flat_df):
    out_df = pd.merge(df, flat_df, on='ticker', how='outer'),
    return out_df

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

print(final := pd.concat([merged_df, name_dummies],axis=1))
final = clean_sneaker_data_for_ml(final)
final.to_csv('final_aj1.csv')
