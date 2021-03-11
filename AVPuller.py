import log
import logging
logger = logging.getLogger('root') 

from alpha_vantage.timeseries import TimeSeries

import os
from os import path
from datetime import datetime
from datetime import timedelta
import time

import re

import pandas as pd
import numpy as np

def slice_generator(years=2, months=12):
    for year in range(1,years+1):
        for month in range(1,months+1):
            yield f"year{year}month{month}"


# TODO pull fundamentals for stock
class AVPuller():
    def __init__(self, key, limit=5, save_dir="data"):
        self.tracker = Tracker(limit)
        self.limit = limit
        self.key = key
        self.save_dir = save_dir

    def meta_data(self):
        data = [re.findall("[\w\d]+",f)[:-1] for f in os.listdir(self.save_dir)]
        meta = pd.DataFrame(data, columns=["ticker","start","end","interval"])
        meta["start"] = pd.to_datetime(meta["start"])
        meta["end"] = pd.to_datetime(meta["end"])
        return meta

    def get_tickers(self):
        ids = []
        for f in os.listdir(self.save_dir):
            split_f = re.split("[-\.]+", f)
            ID = split_f[0]
            if not ID in ["BTC", "ETH"]:
                ids.append(ID)
        ids = list(set(ids))
        ids = sorted(ids)
        return ids

    def pull_tick_monthly(self, tick, adjusted=True):
        self.tracker.wait()
        ts = TimeSeries(key=self.key, output_format='pandas')
        if adjusted:
            df, meta_data = ts.get_monthly_adjusted(tick)
        else:
            df, meta_data = ts.get_monthly(tick)
        df = df.reset_index()
        df['ticker'] = tick
        self.tracker.update(1)
        return df

    def load_data(self, dtype="stock", data_freq="1min"):
        """
            loads all stored data of a particular class 
        """
        files = []
        # find relevant files
        for f in os.listdir(self.save_dir):
            split_f = re.split("[-\.]+", f)
            ID = split_f[0]
            freq = split_f[-2]
            if ID in ["BTC", "ETH"] and dtype == "crypto" and freq == data_freq:
                files.append(f)
            elif dtype == "stock" and freq == data_freq:
                files.append(f)
            else:
                logger.debug(f"file '{f}' did not meet criteria: ({dtype},{data_freq})")
        # read in files into a sigle dataframe  
        return pd.concat([pd.read_csv(path.join(self.save_dir,f),
                                      parse_dates=['date'])
                          for f in files])

    def pull_tickers(self, tickers, freq='15min'):
        """
            freq: '1min','5min','15min','30min','60min'
        """
        ts = TimeSeries(key=self.key, output_format='pandas')
        try:
            frames = []
            for tick in tickers:
                self.tracker.wait()
                df, meta_data = ts.get_intraday(symbol=tick, interval=freq, outputsize='full')
                df = df.reset_index()
                df['ticker'] = tick
                frames.append(df)
                self.tracker.update(1)
        except ValueError as e:
            logger.warning(repr(e))
        return pd.concat(frames)

    def pull_tick_daily(self, tick, adjusted=True):
        self.tracker.wait()
        ts = TimeSeries(key=self.key, output_format='pandas')
        if adjusted:
            df, meta_data = ts.get_daily_adjusted(tick, "full")
        else:
            df, meta_data = ts.get_daily(tick, "full")
        df = df.reset_index()
        df['ticker'] = tick
        self.tracker.update(1)
        return df

    def pull_tick_slice(self, tick, freq, desired_slice, adjusted):
        """
            pulls a slice for a ticker
        """
        logger.info(f"obtaining slice: ({tick},{freq},{desired_slice})")
        ts = TimeSeries(key=self.key, output_format='csv')
        self.tracker.wait()
        reader, meta_data = ts.get_intraday_extended(symbol=tick, interval=freq, slice=desired_slice) # TODO figure out what adjusted isn't accepted https://github.com/RomelTorres/alpha_vantage/blob/develop/alpha_vantage/timeseries.py
        content = [l for l in reader]
        df = pd.DataFrame(content[1:],columns=content[0])
        logger.debug(f"df.columns: {df.columns}")
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename({'time':'date'}, axis=1)
        df['ticker'] = tick
        self.tracker.update(1)
        return df
    
    def pull_tickers_all_slices(self, tickers, freq='15min', adjusted=True):
        """
            pulls all slices for a list of tickers. They're concatenated before being returned
        """
        try:
            frames = [self.pull_tick_all_slices(tick, freq, adjusted) for tick in tickers]
        except ValueError as e:
            logger.warning(repr(e))
        return pd.concat(frames)

    def pull_tick_all_slices(self, ticker, freq='15min', adjusted=True):
        """
            pulls all slices for a particularly ticker. They're concatenated before being returned
        """
        try:
            frames = [self.pull_tick_slice(ticker, freq, desired_slice, adjusted) for desired_slice in slice_generator()]
        except ValueError as e:
            logger.warning(repr(e))
        return pd.concat(frames)

    def pull_cryptos(self, cryptocurrencies, market="USD"):
        cr = CryptoCurrencies(tokens.API_KEY, output_format='pandas')
        try:
            frames = []
            for crypto in cryptocurrencies:
                self.tracker.wait()
                df, meta = cr.get_digital_currency_weekly(crypto, market=market)
                df = df.reset_index()
                df["crypto"] = crypto
                frames.append(df)
                self.tracker.update(1)
        except ValueError as e:
            logger.warning(repr(e))
        return pd.concat(frames)

    def store_as_csv(self, df, crypto_market="USD"):
        # TODO get rid of 'crypto_market' argument
        def format_date(x):
            return x.strftime("%Y%m%d")

        def get_freq(date):
            date = date.sort_values()
            lag_date = date.shift()
            delta = date - lag_date 
            delta_mode = delta.mode()[0]
            if delta_mode == timedelta(minutes=1):
                return "1min"
            elif delta_mode == timedelta(minutes=5):
                return "5min"
            elif delta_mode == timedelta(minutes=15):
                return "15min"
            elif delta_mode == timedelta(minutes=30):
                return "30min"
            elif delta_mode == timedelta(minutes=60):
                return "60min"
            elif delta_mode == timedelta(days=1):
                return "daily"
            elif delta_mode == timedelta(days=7):
                return "weekly"
            elif delta_mode == timedelta(days=31) or delta_mode == timedelta(days=30) or delta_mode == timedelta(days=28):
                return "monthly"
            else:
                raise ValueError(f"Unregistered period of time '{delta_mode}'")


        if "ticker" in df.columns:
            for ticker, _df in df.groupby("ticker"):
                dates = _df['date']
                freq = get_freq(dates)
                f = path.join(self.save_dir,
                            "-".join([ticker,
                                      format_date(min(dates)),
                                      format_date(max(dates)),
                                      freq])+".csv")
                _df.to_csv(f, index=False)
        elif "crypto" in df.columns: 
            for crypto, _df in df.groupby("crypto"):
                dates = _df['date']
                freq = get_freq(dates)
                f = path.join(self.save_dir,
                            "-".join([crypto,
                                    crypto_market,
                                    format_date(min(dates)),
                                    format_date(max(dates)),
                                    freq])+".csv")
                _df.to_csv(f, index=False)


class Tracker():
    def __init__(self, limit=5):
        self.limit = limit
        self.working_list = []

    def update(self, i):
        for _ in range(i):
            self.working_list.append(datetime.now())

    def wait(self, option='until available'):
        now = datetime.now()
        exp = now - timedelta(minutes=5) # time of experation

        l = np.array(self.working_list)
        if option == 'until available':
            isAvailable = False
            while not isAvailable:
                time_elapsed = datetime.now() - l 
                time_left = timedelta(minutes=1) - time_elapsed # 1 minute is API cooldown time
                active = time_left[time_left > timedelta(minutes=0)]
                logger.debug(f"active: {active}")
                if len(active) >= self.limit:
                    sleep_time = min(active).total_seconds()
                    logger.info(f"sleeping for {sleep_time}...")
                    time.sleep(sleep_time)
                    logger.info("awake.")
                else:
                    isAvailable = True
        else:
            raise ValueError(f"option '{option}' not implemented.")
        return isAvailable
