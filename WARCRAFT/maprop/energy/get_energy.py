# -*- coding: utf-8 -*-

"""
Reading in the data, sklearn style...

The data contains the following column:

#DateTime Holiday HolidayFlag DayOfWeek WeekOfYear Day Month Year PeriodOfDay Fo recastWindProduction SystemLoadEA SMPEA ORKTemperature ORKWindspeed CO2Intensity ActualWindProduction SystemLoadEP2 SMPEP2

#DateTime and Holiday: is a string and subsumed by following features HolidayFlag: is Boolean and identicaly for each day DayOfWeek WeekOfYear Day Month Year: is discrete and identicaly for each day PeriodOfDay: is discrete 0..47 ORKTemperature ORKWindspeed: contains NAN and are questionable (actual values) ActualWindProduction SystemLoadEP2 SMPEP2: Actual values, with SMPEP2 the label
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# prep numpy arrays, Xs will contain groupID as first column
def get_energy(fname=None, trainTestRatio=0.70):
    df = get_energy_pandas(fname)

    length = df['groupID'].nunique()
    grouplength = 48

    # numpy arrays, X contains groupID as first column
    X1g = df.loc[:, df.columns != 'SMPEP2'].values
    y = df.loc[:, 'SMPEP2'].values

    #no negative values allowed...for now I just clamp these values to zero. They occur three times in the training data.
    # for i in range(len(y)):
    #     y[i] = max(y[i], 0)


    # ordered split per complete group
    train_len = int(trainTestRatio*length)

    # the splitting
    X_1gtrain = X1g[:grouplength*train_len]
    y_train = y[:grouplength*train_len]
    X_1gtest  = X1g[grouplength*train_len:]
    y_test  = y[grouplength*train_len:]
    
    

    #print(len(X1g_train),len(X1g_test),len(X),len(X1g_train)+len(X1g_test))
    return (X_1gtrain, y_train, X_1gtest, y_test)


def get_energy_grouped(fname=None):
    df = get_energy_pandas(fname)

    # put the 'y's into columns (I hope this respects the ordering!)
    t = df.groupby('groupID')['SMPEP2'].apply(np.array)
    grpY = np.vstack(t.values) # stack into a 2D array
    # now something similar but for the features... lets naively just take averages
    grpX = df.loc[:, df.columns != 'SMPEP2'].groupby('groupID').mean().as_matrix()

    # train/test splitting, sklearn is so convenient
    (grpX_train, grpX_test, grpY_train, grpY_test) = \
        train_test_split(grpX, grpY, test_size=0.3, shuffle=False)

    return (grpX_train, grpY_train, grpX_test, grpY_test)


def get_energy_pandas(fname=None):
    if fname == None:
        fname = "prices2013.dat"

    df = pd.read_csv(fname, delim_whitespace=True, quotechar='"')
    # remove unnecessary columns
    df.drop(['#DateTime', 'Holiday', 'ActualWindProduction', 'SystemLoadEP2'], axis=1, inplace=True)
    # remove columns with missing values
    df.drop(['ORKTemperature', 'ORKWindspeed'], axis=1, inplace=True)

    # missing value treatment
    # df[pd.isnull(df).any(axis=1)]
    # impute missing CO2 intensities linearly
    df.loc[df.loc[:,'CO2Intensity'] == 0, 'CO2Intensity'] = np.nan # an odity
    df.loc[:,'CO2Intensity'].interpolate(inplace=True)
    # remove remaining 3 days with missing values
    grouplength = 48
    for i in range(0, len(df), grouplength):
        day_has_nan = pd.isnull(df.loc[i:i+(grouplength-1)]).any(axis=1).any()
        if day_has_nan:
            #print("Dropping",i)
            df.drop(range(i,i+grouplength), inplace=True)
    # data is sorted by year, month, day, periodofday; don't want learning over this
    df.drop(['Day', 'Year', 'PeriodOfDay'], axis=1, inplace=True)

    # insert group identifier at beginning
    grouplength = 48
    length = int(len(df)/48) # 792
    gids = [gid for gid in range(length) for i in range(grouplength)]
    df.insert(0, 'groupID', gids)

    return df



if __name__ == '__main__':
    df = get_energy_pandas()
    print(df.head())

    (X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
    print([len(x) for x in (X_1gtrain, y_train, X_1gtest, y_test)])

    ### Options to try for learning:
    # split DayOfWeek into Weekday/Weekend, perhaps even split up days
    # split up Month into seasons
    # do use ORK*s but with missing value imputation
    # remove WeekOfYear?
