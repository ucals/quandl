import pandas as pd
import quandl
import os


quandl.ApiConfig.api_key = os.getenv('QUANDL_APY_KEY')


def get_djia_symbols():
    data = pd.read_html("https://www.cnbc.com/dow-30/")
    table = data[0]
    return table['Symbol'].tolist()


def get_sp500_symbols():
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = data[0]
    return table['Symbol'].tolist()


def get_sf1_symbols():
    data = quandl.get_table('SHARADAR/TICKERS', table='SF1', qopts={"columns": ['ticker']}, paginate=True)
    return data['ticker'].tolist()


def download_sf1_data():
    fname = os.path.join(os.getcwd(), 'data_defensive', 'sf1.csv')

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    list_sf1 = get_sf1_symbols()
    list_dfs = []
    i = 1
    for x in batch(list_sf1, 200):
        print(f'Batch {i}: from {x[0]} to {x[-1]}')
        columns = ['ticker', 'calendardate', 'datekey', 'revenueusd', 'currentratio', 'eps', 'dps', 'bvps', 'price']
        df = quandl.get_table('SHARADAR/SF1', dimension='ARY', ticker=x, qopts={"columns":columns},
                              paginate=True)
        df.sort_values(['ticker', 'calendardate'], inplace=True)
        list_dfs.append(df)
        i += 1

    final_df = pd.concat(list_dfs, axis=0)
    final_df.to_csv(fname, index=False)