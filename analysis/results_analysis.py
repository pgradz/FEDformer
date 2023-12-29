import os
import pandas as pd
import numpy as np
import logging


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_h = logging.StreamHandler()
    stream_h.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    stream_h.setFormatter(formatter)
    logger.addHandler(stream_h)

    return logger

logger = get_logger()


def backtest(df, lower_bound, upper_bound, results_folder, seed, task_id, args):
    ''' This function is used to backtest the model
    args:
        df: dataframe with predictions and actual values
        lower_bound: lower bound for the prediction
        upper_bound: upper bound for the prediction
    return:
    '''
    df.reset_index(inplace=True, drop=True)
    conditions = [
        (df['prediction'] <= lower_bound),
        (df['prediction'] > lower_bound) & (df['prediction'] <= upper_bound),
        (df['prediction'] > upper_bound)
    ]
    values = [0, 0.5, 1]
    df['trade'] = np.select(conditions, values)

    cols = ['datetime','open', 'close', 'high','low', 'cost', 'trade', 'prediction', 'prc_change']
    if 'triple_barrier' in args.run_subtype:
        cols = cols + ['time_step','barrier_touched', 'barrier_touched_date', 'bottom_barrier', 'top_barrier']
    else:
        df['prc_change'] = df['y_pred'] - 1
    df = df[cols]

    if 'triple_barrier' in args.run_subtype:
        df = run_trades_3_barriers(df)
    else:
        df = run_trades_one_step(df)


    df.to_csv(os.path.join(results_folder,
                            f'''backtest_{task_id}_{seed}.csv'''))
    # SUMMARIZE RESULTS
    hits = df.loc[((df['transaction'] == 'buy') & (df['prc_change'] > 0)) |
                        ((df['transaction'] == 'sell') & (df['prc_change'] < 0))].shape[0]
    transactions = df.loc[df['transaction'].isin(['buy', 'sell'])].shape[0]
    try:
        hits_ratio = hits / transactions
    except ZeroDivisionError:
        hits_ratio = 0

    logger.info('Hit ratio:  %.2f on %.0f trades', hits_ratio, transactions)

def run_trades_3_barriers(df):
        '''
        This function is used to run trades based on 3 barriers. For this method, transaction costs
        happen always to open and close. There is no leaving the position open as trading is based on
        take profit and stop loss. Note: this approximate result. Exact implementation is in PerfromancEvaluator.
        args:
            df: dataframe with predictions and actual values
        return: 
            df: dataframe with budget and transaction columns
        '''
         # INITIALIZE PORTFOLIO
        budget = 100
        logger.info('Starting trading with:  %.2f', budget)
        transaction = 'No trade'
        i = 0
        while i < df.shape[0]:
        
            if df.loc[i, 'trade'] == 1:
                if transaction != 'No trade':
                    # first close open position if open
                    budget = budget * (1 - df.loc[i, 'cost']) 
                # then open new position
                budget = budget * (1 - df.loc[i, 'cost'])
                transaction = 'buy'
                budget = budget + budget * df.loc[i, 'prc_change']
                df.loc[i, 'budget'] = budget
                df.loc[i, 'transaction'] = transaction
                i = i + df.loc[i, 'time_step'] # jump till barrier is touched
            elif df.loc[i, 'trade'] == 0:
                # add transaction cost if position changes
                if transaction != 'No trade':
                    # first close open position if open
                    budget = budget * (1 - df.loc[i, 'cost']) 
                 # then open new position
                budget = budget * (1 - df.loc[i, 'cost'])
                transaction = 'sell'
                budget = budget + budget * (-df.loc[i, 'prc_change'])
                df.loc[i, 'budget'] = budget
                df.loc[i, 'transaction'] = transaction
                i = i + df.loc[i, 'time_step'] # jump till barrier is touched
            elif df.loc[i, 'trade'] == 0.5:
                if transaction in ['buy', 'sell']:
                    budget = budget * (1 - df.loc[i, 'cost']) # add cost while closing position
                    transaction = 'No trade'
                df.loc[i, 'budget'] = budget
                df.loc[i, 'transaction'] = transaction
                i = i + 1
        # close any open transaction at the end of the test set        
        if transaction != 'No trade':
            budget = budget * (1 - df.loc[df.shape[0]-1, 'cost'])
                
        logger.info('Final budget:  %.2f', budget)
        return df

def run_trades_one_step(df):
    '''
    This function is used to run trades based on step ahead predictions. For this method, transaction costs
    happen sometimes can be avoided if position is not changed.
    args:
        df: dataframe with predictions and actual values
    return: 
        df: dataframe with budget and transaction columns
    '''
     # INITIALIZE PORTFOLIO
    budget = 100
    logger.info('Starting trading with:  %.2f', budget)
    transaction = 'No trade'
    i = 0
    while i < df.shape[0]:
    
        if df.loc[i, 'trade'] == 1:
            if transaction == 'sell':
                # first close open short position if open
                budget = budget * (1 - df.loc[i, 'cost']) 
            if transaction != 'buy':
                # then open new position if needed
                budget = budget * (1 - df.loc[i, 'cost'])
            transaction = 'buy'
            budget = budget + budget * df.loc[i, 'prc_change']
            df.loc[i, 'budget'] = budget
            df.loc[i, 'transaction'] = transaction
            i+=1 # jump one step ahead
        elif df.loc[i, 'trade'] == 0:
            # add transaction cost if position changes
            if transaction == 'buy':
                # first close open long position if open
                budget = budget * (1 - df.loc[i, 'cost']) 
            if transaction != 'sell':
                # then open new position if needed
                budget = budget * (1 - df.loc[i, 'cost'])
            transaction = 'sell'
            budget = budget + budget * (-df.loc[i, 'prc_change'])
            df.loc[i, 'budget'] = budget
            df.loc[i, 'transaction'] = transaction
            i+=1 # jump one step ahead
        elif df.loc[i, 'trade'] == 0.5:
            if transaction in ['buy', 'sell']:
                budget = budget * (1 - df.loc[i, 'cost']) # add cost while closing position
                transaction = 'No trade'
            df.loc[i, 'budget'] = budget
            df.loc[i, 'transaction'] = transaction
            i+=1
    # close any open transaction at the end of the test set        
    if transaction != 'No trade':
        budget = budget * (1 - df.loc[df.shape[0]-1, 'cost'])
    logger.info('Final budget:  %.2f', budget)
    return df

def load_backtest_data(exp, results_folder):
            result_files  = [os.path.join(results_folder, file) for file in os.listdir(results_folder) if exp in file] 
            for file in result_files:
                yield pd.read_csv(file, parse_dates=['time_stamps'])

def run_analysis(args, setting):

    seeds = {'Exp_0': 12345, 'Exp_1': 123456, 'Exp_2': 1234567}
    results_folder = './results/' + setting + '/'
    # results_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/FEDformer_results/BTCUSDT_720min_FEDformer_random_modes64_cryptoh1_ftM_sl96_ll48_pl1_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue'
    experiments = ['Exp_0', 'Exp_1', 'Exp_2']
    for exp in experiments:
   
        seed = seeds[exp]
        results = pd.concat(load_backtest_data(exp, results_folder))
        results.sort_values(by=['time_stamps'], inplace=True)
        results.rename(columns={'time_stamps': 'datetime', 'preds':'prediction'}, inplace=True)
        results.drop(['y_pred'], axis=1, inplace=True)
        raw_data = pd.read_csv(os.path.join(args.root_path, args.data_path), parse_dates=['datetime'])
        if 'triple_barrier' in args.run_subtype:
            results = results.merge(raw_data[['datetime','open', 'close', 'high','low','y_pred','time_step',
                                              'barrier_touched', 'barrier_touched_date', 'bottom_barrier', 'top_barrier', 'prc_change']],
                                                on='datetime', how='left')
        else:
            results = results.merge(raw_data[['datetime','open', 'close', 'high','low','y_pred']], on='datetime', how='left')
        results['cost'] = 0.001
        backtest(results, 0.4,0.6, results_folder, seed, args.task_id, args)


    