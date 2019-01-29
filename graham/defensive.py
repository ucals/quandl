import sys
import pandas as pd
import numpy as np
import quandl
import os
import warnings
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
import humanize

warnings.filterwarnings('ignore')
quandl.ApiConfig.api_key = os.getenv('QUANDL_APY_KEY')


def get_data(ticker, dt=datetime.today(), force_download=False):
    fname = os.path.join(os.getcwd(), 'data_defensive', 'sf1.csv')
    columns = ['calendardate', 'datekey', 'revenueusd', 'currentratio', 'eps', 'dps', 'bvps', 'price']
    if force_download or not os.path.isfile(fname):
        df = quandl.get_table('SHARADAR/SF1', dimension='ARY', ticker=ticker, qopts={"columns": columns})
        df.sort_values('calendardate', inplace=True)
    else:
        df = pd.read_csv(fname, parse_dates=['calendardate', 'datekey'])
        df = df[df['ticker'] == ticker][columns]

    df = df[df['datekey'] <= dt]
    df = df.groupby('calendardate').tail(1)
    return df


def criteria_defensive_investor(ticker_or_list, relax_current_ratio=False, verbose=False, dt=datetime.today(),
                                force_download=False, show_progress=True, return_list_five=False):
    return_list = []    # if return_list_five is True, returns only the list of companies passing first 5 criteria;
    # otherwise, returns a full dataframe with data from all companies passed in ticker_or_list

    # The function accepts either a ticker of a list of tickers. This block treats this
    list_tickers = []
    if isinstance(ticker_or_list, str):
        list_tickers.append(ticker_or_list)
    else:
        list_tickers = ticker_or_list

    # Creates the empty dataframe, to be returned in case of return_list_five == False
    summary_columns = ['ticker', 'last_date', 'first_date', 'revenueusd', 'current_ratio',
                       'positive_eps_p10yrs_count', 'dividend_distribution_p20yrs_count',
                       'earnings_change_p10yrs', 'pe', 'pb', 'pexpb', 'size_criteria',
                       'financial_condition_criteria', 'earnings_stability_criteria',
                       'dividend_record_criteria', 'earnings_growth_criteria', 'pe_criteria',
                       'pb_criteria', 'first_five_criteria', 'full_criteria']
    df_ = pd.DataFrame(columns=summary_columns)

    # If show_progress == True, the iterator show a progress bar. Otherwise, shows nothing. This block treats this
    def empty_func(x):
        return x

    if show_progress:
        func = tqdm
    else:
        func = empty_func

    # Main iterator
    for ticker in func(list_tickers):
        if verbose:
            print('\nTest for ' + ticker + ':')
        data = get_data(ticker, dt=dt, force_download=force_download)

        # In case of no data
        if data.shape[0] == 0:  # No data
            if verbose:
                print('- No data available')
            df_ = df_.append(pd.Series([float('NaN')] * len(summary_columns), index=summary_columns), ignore_index=True)
            df_.at[df_.index[-1], 'ticker'] = ticker
            continue

        # Size criteria
        size_criteria = data['revenueusd'].values[-1] > 100000000
        if return_list_five and not size_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Financial condition criteria
        # TODO: parameter relax_current_ratio should be replaced by a better way to treat current ratio of financial
        # companies
        if (data['currentratio'].values[-1] is not None) and (not np.isnan(data['currentratio'].values[-1])):
            current_ratio = data['currentratio'].values[-1]
            financial_condition_criteria = current_ratio > 2
        else:
            current_ratio = float('NaN')
            financial_condition_criteria = relax_current_ratio

        if return_list_five and not financial_condition_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Earnings stability criteria
        earnings_stability_criteria = (data['eps'].tail(10) > 0).all()
        if return_list_five and not earnings_stability_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Dividends record criteria
        dividend_record_criteria = (data['dps'].tail(20) > 0).all()
        if return_list_five and not dividend_record_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Earnings growth criteria
        last_year = pd.to_datetime(data['calendardate'].values[-1]).year
        eps_0 = data[(data['calendardate'].dt.year > last_year - 13) &
                     (data['calendardate'].dt.year <= last_year - 10)]['eps'].mean()
        eps_1 = data[(data['calendardate'].dt.year > last_year - 3) &
                     (data['calendardate'].dt.year <= last_year)]['eps'].mean()
        earnings_growth_criteria = (np.float64(eps_1) / eps_0) > 1.33
        if return_list_five and not earnings_growth_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Only first five criteria
        first_five_criteria = size_criteria and financial_condition_criteria and earnings_stability_criteria \
                              and dividend_record_criteria and earnings_growth_criteria

        # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
        if return_list_five and not first_five_criteria:
            continue
        # If is just to return the list of companies passing first 5 criteria, and success, adds to list AND goes to next iteration
        elif return_list_five and first_five_criteria:
            return_list.append(ticker)
            continue

        # P/E ratio criteria
        current_price = data['price'].values[-1]
        pe = current_price / eps_1
        pe_criteria = pe < 15

        # Price to Assets criteria
        pb = current_price / data['bvps'].values[-1]
        pb_criteria = pb < 1.5
        if (pe * pb < 22.5):
            pb_criteria = True

        # Full criteria
        full_criteria = size_criteria and financial_condition_criteria and earnings_stability_criteria \
                        and dividend_record_criteria and earnings_growth_criteria and pe_criteria and pb_criteria

        # Add to dataframe
        my_dic = {'ticker': ticker,
                  'last_date': data['calendardate'].values[-1],
                  'first_date': data['calendardate'].values[0],
                  'revenueusd': data['revenueusd'].values[-1],
                  'current_ratio': current_ratio,
                  'positive_eps_p10yrs_count': data.tail(10)[data['eps'] > 0]['eps'].count(),
                  'dividend_distribution_p20yrs_count': data.tail(20)[data['dps'] > 0]['dps'].count(),
                  'earnings_change_p10yrs': (np.float64(eps_1) / eps_0),
                  'pe': pe,
                  'pb': pb,
                  'pexpb': pe * pb,
                  'size_criteria': size_criteria,
                  'financial_condition_criteria': financial_condition_criteria,
                  'earnings_stability_criteria': earnings_stability_criteria,
                  'dividend_record_criteria': dividend_record_criteria,
                  'earnings_growth_criteria': earnings_growth_criteria,
                  'pe_criteria': pe_criteria,
                  'pb_criteria': pb_criteria,
                  'first_five_criteria': first_five_criteria,
                  'full_criteria': full_criteria}
        df_.loc[len(df_)] = my_dic

        if verbose:
            print('- Size criteria: \t\t' + str(size_criteria) +
                  '\tRevenues of $' + humanize.intword(data['revenueusd'].values[-1]) + ' (threshold is $100 million)')
            print('- Financial condition criteria: ' + str(financial_condition_criteria) +
                  '\tCurrent ratio of %1.2f' % current_ratio + ' (threshold is 2.0)')
            print('- Earnings stability criteria: \t' + str(earnings_stability_criteria) +
                  '\tPositive earnings in %d of past 10 years' % data.tail(10)[data['eps'] > 0]['eps'].count())
            print('- Dividend record criteria: \t' + str(dividend_record_criteria) +
                  '\tDistribution of dividend in %d of past 20 years' % data.tail(20)[data['dps'] > 0]['dps'].count())
            print('- Earnings growth criteria: \t' + str(earnings_growth_criteria) +
                  '\tEarnings change of %+.0f%%' % (100 * ((np.float64(eps_1) / eps_0) - 1)) + ' in past 10 years (minimum is +33%)')
            print('- Moderate P/E ratio criteria: \t' + str(pe_criteria) +
                  '\tCurrent price is %1.1fx avg P3yrs earnings (limit is 15)' % pe)
            print('- Moderate P/B ratio criteria: \t' + str(pb_criteria) +
                  '\tCurrent price is %1.1fx last book value (limit 1.5), \n\t\t\t\t\tand PE * PB is %1.1f (limit 22.5)' % (pb, pe * pb))
            print('- Full criteria: \t\t' + str(full_criteria))

    if return_list_five:
        return return_list
    else:
        return df_


def analysis(df):
    c1 = len(df[df['size_criteria'] == True]['ticker'].tolist())
    c2 = len(df[df['financial_condition_criteria'] == True]['ticker'].tolist())
    c3 = len(df[df['earnings_stability_criteria'] == True]['ticker'].tolist())
    c4 = len(df[df['dividend_record_criteria'] == True]['ticker'].tolist())
    c5 = len(df[df['earnings_growth_criteria'] == True]['ticker'].tolist())
    c6 = len(df[df['pe_criteria'] == True]['ticker'].tolist())
    c7 = len(df[df['pb_criteria'] == True]['ticker'].tolist())

    c8 = len(df[(df['size_criteria'] == True) &
                (df['financial_condition_criteria'] == True) &
                (df['earnings_stability_criteria'] == True) &
                (df['dividend_record_criteria'] == True) &
                (df['earnings_growth_criteria'] == True)]['ticker'].tolist())

    cdi = len(df[df['full_criteria'] == True]['ticker'].tolist())

    print(f'{c1:4d} companies passing size criteria of minimum $100 million revenues')
    print(f'{c2:4d} companies passing financial condition criteria of minimum 2 current ratio')
    print(f'{c3:4d} companies passing earnings stability criteria of positive earnings in past 10 years')
    print(f'{c4:4d} companies passing dividend record criteria of uninterrupted payments in past 20 years')
    print(f'{c5:4d} companies passing earnings growth criteria of minimum 33% growth in past 10 years')
    print(f'{c6:4d} companies passing moderate PE ratio criteria of maximum 15')
    print(f'{c7:4d} companies passing moderate PB ratio criteria of maximum 1.5, or PE * PB of maximum 22.5')
    print(f'{c8:4d} companies passing all except moderate PE and PB ratio criteria')
    print(f'{cdi:4d} companies passing all criteria')

    df['count_successful_criteria'] = df.apply(lambda row: sum([row['size_criteria'],
                                                                row['financial_condition_criteria'],
                                                                row['earnings_stability_criteria'],
                                                                row['dividend_record_criteria'],
                                                                row['earnings_growth_criteria'],
                                                                row['pe_criteria'],
                                                                row['pb_criteria']]), axis=1)

    df['count_successful_criteria_except_pepb'] = df.apply(lambda row: sum([row['size_criteria'],
                                                                            row['financial_condition_criteria'],
                                                                            row['earnings_stability_criteria'],
                                                                            row['dividend_record_criteria'],
                                                                            row['earnings_growth_criteria']]), axis=1)

    return df


def criteria_defensive_investor_list(ticker_or_list, all_data=None, relax_current_ratio=False, dt=datetime.today(),
                                     show_progress=True):
    return_list = []

    # The function accepts either a ticker of a list of tickers. This block treats this
    list_tickers = []
    if isinstance(ticker_or_list, str):
        list_tickers.append(ticker_or_list)
    else:
        list_tickers = ticker_or_list

    # If show_progress == True, the iterator show a progress bar. Otherwise, shows nothing. This block treats this
    def empty_func(x):
        return x

    if show_progress:
        func = tqdm
    else:
        func = empty_func

    # Get all data at once
    if all_data is None:
        fname = os.path.join(os.getcwd(), 'data_defensive', 'sf1.csv')
        df_all = pd.read_csv(fname, parse_dates=['calendardate', 'datekey'])
    else:
        df_all = all_data

    # Main iterator
    for ticker in func(list_tickers):
        data = df_all[(df_all['ticker'] == ticker) & (df_all['datekey'] <= dt)]
        data = data.groupby('calendardate').tail(1)

        # In case of no data
        if data.shape[0] == 0:  # No data
            continue

        # Size criteria
        size_criteria = data['revenueusd'].values[-1] > 100000000
        if not size_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Financial condition criteria
        # TODO: parameter relax_current_ratio should be replaced by a better way to treat current ratio of financial
        # companies
        if (data['currentratio'].values[-1] is not None) and (not np.isnan(data['currentratio'].values[-1])):
            current_ratio = data['currentratio'].values[-1]
            financial_condition_criteria = current_ratio > 2
        else:
            current_ratio = float('NaN')
            financial_condition_criteria = relax_current_ratio

        if not financial_condition_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Earnings stability criteria
        earnings_stability_criteria = (data['eps'].tail(10) > 0).all()
        if not earnings_stability_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Dividends record criteria
        dividend_record_criteria = (data['dps'].tail(20) > 0).all()
        if not dividend_record_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Earnings growth criteria
        last_year = pd.to_datetime(data['calendardate'].values[-1]).year
        eps_0 = data[(data['calendardate'].dt.year > last_year - 13) &
                     (data['calendardate'].dt.year <= last_year - 10)]['eps'].mean()
        eps_1 = data[(data['calendardate'].dt.year > last_year - 3) &
                     (data['calendardate'].dt.year <= last_year)]['eps'].mean()
        earnings_growth_criteria = (np.float64(eps_1) / eps_0) > 1.33
        if not earnings_growth_criteria:  # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
            continue

        # Only first five criteria
        first_five_criteria = size_criteria and financial_condition_criteria and earnings_stability_criteria \
                              and dividend_record_criteria and earnings_growth_criteria

        # If is just to return the list of companies passing first 5 criteria, and failed, goes to next iteration
        if not first_five_criteria:
            continue

        return_list.append(ticker)

    return return_list
