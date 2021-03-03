import pandas as pd
import argparse

def clean_sneaker_data_for_ml(df, flat_df)
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

def merge_df(df, flat_df):
    pass

parser = argparse.ArgumentParser(description='Auto-crop a directory of images')
parser.add_argument('-df','--dataframe', required=True, help='dataframe containing scraped shoe data')
parser.add_argument('-fdf','--flat_dataframe', required=True, help='dataframe containing image data')

args = parser.parse_args(sys.argv[1:])


df = load_df(args.dataframe)
flat_df = load_df(args.flat_dataframe)
