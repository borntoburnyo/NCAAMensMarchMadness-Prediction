# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list the files in the input directory

import os
print(os.listdir("../input"))

# load interested dataset 
# input some data of interest 
tour_seed = pd.read_csv("../input/NCAATourneySeeds_SampleTourney2018.csv") # 1985-2018
tour_comp_result = pd.read_csv("../input/NCAATourneyCompactResults.csv") # 1985-2017
reg_detailed_result = pd.read_csv("../input/RegularSeasonDetailedResults_Prelim2018.csv") # 2003-2018
reg_comp_result = pd.read_csv("../input/RegularSeasonCompactResults_Prelim2018.csv") # 1985-2018
sub1 = pd.read_csv("../input/SampleSubmissionStage1.csv") # 2014-2017
sub2 = pd.read_csv("../input/SampleSubmissionStage2.csv") # 2018
coaches = pd.read_csv("../input/TeamCoaches_Prelim2018.csv") # 1985-2018



# EDA and Feature Engineering

"""
How often do teams enter Division I for all teams in 2018? 

For the past 32 seasons (1985-2017), more than half of the team that 
entered Division I League in 2018 has achieved this for more than 10 times, 
25% of team has achieved more than 19 times.
"""

df_entry = pd.merge(tour_seed.loc[tour_seed.Season < 2018], tour_seed[tour_seed.Season == 2018], how="right", on="TeamID")
df_entry = df_entry.groupby("TeamID", as_index=False).aggregate({"Season_x": "count"})
df_entry.rename({"Season_x": "Entry"}, axis="columns", inplace=True)
df_entry.sort_values(by="Entry", inplace=True)
df_entry.reset_index(drop=True, inplace=True)

plt.figure(figsize=(16,8))
ax = sns.barplot(df_entry.index, "Entry", data=df_entry, palette="Blues_d")
ax.set_xticklabels(labels=df_entry.TeamID, rotation=45)
for p, entry in zip(ax.patches, df_entry.Entry):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, 
           height,
           entry,
           ha = "center")
plt.xlabel("Team ID")
plt.title("# of previous entry to Division I before 2018")
plt.show()

print(df_entry.Entry.describe())


"""
Functions to get coaches' life time regular season and tournament winning rate
Which I think would be impoart features 
"""


# find regular season winning rate
def coach_reg_win_rate(season):
    """ 
    season: INT
        Represents season to be predicted. 
    return: DataFrame with 'TeamID' and 'CoachRegWinRate' as columns
    """
    df_reg_win = pd.merge(reg_comp_result[reg_comp_result.Season <= season], 
                          coaches, how="inner", left_on=["Season", "WTeamID"], right_on=["Season","TeamID"])
    df_reg_lose = pd.merge(reg_comp_result[reg_comp_result.Season <= season], 
                           coaches, how="inner", left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    
    reg_win_num = df_reg_win.groupby("CoachName", as_index=False).aggregate({"Season":"count"})
    reg_win_num.rename({"Season":"NumWins"}, axis="columns", inplace=True)
    reg_lose_num = df_reg_lose.groupby("CoachName", as_index=False).aggregate({"Season":"count"})
    reg_lose_num.rename({"Season":"NumLoses"}, axis="columns", inplace=True)
    
    reg_concat = pd.merge(reg_win_num, reg_lose_num, how="inner", on="CoachName")
    reg_concat["CoachRegWinRate"] = reg_concat.NumWins / (reg_concat.NumWins + reg_concat.NumLoses)
    
    # If there are multiple coaches for single team, choose the last coach is final one for that team
    if season == 2018:
        tour_coaches = pd.merge(tour_seed[tour_seed.Season==season], 
                                coaches[coaches["LastDayNum"]==115], left_on=["Season", "TeamID"], 
                                right_on=["Season", "TeamID"])
    else:
        tour_coaches = pd.merge(tour_seed[tour_seed.Season==season], 
                                coaches[coaches["LastDayNum"]==154], left_on=["Season", "TeamID"], 
                                right_on=["Season", "TeamID"])
    
    reg_coaches_win_rate = reg_concat.merge(tour_coaches, on="CoachName")
    return reg_coaches_win_rate.loc[:, ["Season", "TeamID", "CoachRegWinRate"]]



def coach_tour_win_rate(season):
    """
    This is actually all tour win rate before current year that need to be predicted.
    
    season: INT
        Represents season to be predicted
    return: DataFrame with 'TeamID' and 'CoachTourWinRate' as columns
    """
    df_tour_win = pd.merge(tour_comp_result.loc[tour_comp_result["Season"] < season], 
                           coaches, how="inner", 
                           left_on=["Season", "WTeamID"], 
                           right_on=["Season","TeamID"])
    df_tour_lose = pd.merge(tour_comp_result.loc[tour_comp_result["Season"] < season], 
                            coaches, how="inner", 
                            left_on=["Season", "LTeamID"], 
                            right_on=["Season", "TeamID"])

    tour_win_num = df_tour_win.groupby("CoachName", as_index=False).aggregate({"Season":"count"})
    tour_win_num.rename({"Season":"NumWins"}, axis="columns", inplace=True)
    tour_lose_num = df_tour_lose.groupby("CoachName", as_index=False).aggregate({"Season":"count"})
    tour_lose_num.rename({"Season":"NumLoses"}, axis="columns", inplace=True)
    tour_concat = pd.merge(tour_win_num, tour_lose_num, how="inner", on="CoachName")
    tour_concat["CoachTourWinRate"] = tour_concat.NumWins / (tour_concat.NumWins + tour_concat.NumLoses)
    
    # If there are multiple coaches for single team, choose the last coach is final one for that team
    if season == 2018:
        tour_coaches = pd.merge(tour_seed[tour_seed.Season==season], 
                                coaches[coaches["LastDayNum"]==115], 
                                left_on=["Season", "TeamID"], 
                                right_on=["Season", "TeamID"])
    else:
        tour_coaches = pd.merge(tour_seed[tour_seed.Season==season], 
                                coaches[coaches["LastDayNum"]==154], 
                                left_on=["Season", "TeamID"], 
                                right_on=["Season", "TeamID"])
    
    tour_coaches_win_rate = tour_concat.merge(tour_coaches,how="right", on="CoachName")
   
    # when coaches has not tournament data, fill winning rate with 0.5 as non-informative guess
    tour_coaches_win_rate.fillna(value=0.5, inplace=True) 
    return tour_coaches_win_rate.loc[:, ["Season", "TeamID", "CoachTourWinRate"]]



"""
Some coaches has high lifetime regular season winning rate, 
but low tournament winning rate, other keep consistent, 
no matter low or high.
"""

reg2018 = coach_reg_win_rate(2018)
tour2018 = coach_tour_win_rate(2018)
rate2018 = reg2018.merge(tour2018, on="TeamID")

plt.figure(figsize=(16,8))
plt.plot(np.arange(len(rate2018)), rate2018.CoachRegWinRate)
plt.plot(np.arange(len(rate2018)), rate2018.CoachTourWinRate)
plt.legend()
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel("Team Sequences")
plt.ylabel("Win Rate")
plt.title("Coach Lifetime Regular / Tournament Win Rate for Year 2018")
plt.show()

"""
Now write funtion to get regualar season stats of each team 
Where the data is only for one specific year, as I think previous 
years' data won't be representative, student players get graduate
or program transfer every year
"""

win_cols = ['WFGM',
'WFGA',
'WFGM3',
'WFGA3',
'WFTM',
'WFTA',
'WOR',
'WDR',
'WAst',
'WTO',
'WStl',
'WBlk',
'WPF']

lose_cols = ['LFGM',
'LFGA',
'LFGM3',
'LFGA3',
'LFTM',
'LFTA',
'LOR',
'LDR',
'LAst',
'LTO',
'LStl',
'LBlk',
'LPF']

print("(win cols length, lose cols length) = {}".format((len(win_cols), len(lose_cols))))


def reg_stats(season):
    """
    season: INT 
        Season to be predicted
    Return: DataFrame of regular season stats 
    """
    tour_team = tour_seed[tour_seed.Season == season]
    reg_win = reg_detailed_result.merge(tour_team, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
    reg_lose = reg_detailed_result.merge(tour_team, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    
    win_games = reg_win.groupby("WTeamID", as_index=False).size().reset_index(name="WGamePlayed")
    reg_win_sum = reg_win.groupby("WTeamID", as_index=False)[win_cols].sum()
    reg_win_sum = reg_win_sum.merge(win_games, on="WTeamID")
    reg_win_sum.rename(columns = lambda x: x[1:], inplace=True) # get rid of "W" mark
    
    lose_games = reg_win.groupby("LTeamID", as_index=False).size().reset_index(name="LGamePlayed")
    reg_lose_sum = reg_lose.groupby("LTeamID", as_index=False)[lose_cols].sum()
    reg_lose_sum = reg_lose_sum.merge(lose_games, on="LTeamID")
    reg_lose_sum.rename(columns = lambda x: x[1:], inplace=True) # get rid of "L" mark
    
    reg_sum = pd.concat((reg_win_sum, reg_lose_sum)).sort_values(by="TeamID")
    reg_sum = reg_sum.groupby("TeamID", as_index=False)[[x[1:] for x in win_cols] + ["GamePlayed"]].sum()
    
    reg_sum["FGR"] = reg_sum["FGM"] / reg_sum["FGA"]
    reg_sum["FGR3"] = reg_sum["FGM3"] / reg_sum["FGA3"]
    reg_sum["FTR"] = reg_sum["FTM"] / reg_sum["FTA"]
    
    
    for i in ["OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]:
        reg_sum[i+"R"] = reg_sum[i] / reg_sum["GamePlayed"]
    
    reg_sum.insert(loc=0, column="Season", value=season)
    
    return reg_sum

# Now define function to combine all feature together 

def concat_stats(seasons):
    """
    years: List of int
        A list of years for which data to be provided
    Return: DataFrame 
        Concated stats for list of years
    """
    stats_df = pd.DataFrame()
    for season in seasons:
        reg_win_rate = coach_reg_win_rate(season)
        tour_win_rate = coach_tour_win_rate(season)
        reg_stats_df = reg_stats(season)
        
        temp_df = pd.merge(reg_win_rate, tour_win_rate, on=["Season", "TeamID"])
        temp_df = temp_df.merge(reg_stats_df, on=["Season", "TeamID"])
        
        stats_df = pd.concat((stats_df, temp_df))
    
    stats_df.reset_index(drop=True, inplace=True)
    
    return stats_df

"""
After obtain all these feature engineering functions
Let's make our training dataset 
"""

def seed_to_int(seed):
    return int(''.join(filter(lambda x: x.isdigit(), seed)))

# change seed to int 
tour_seed["SeedINT"] = tour_seed.Seed.apply(seed_to_int)

#train data 2003 - 2017
tour_train = tour_comp_result.loc[tour_comp_result["Season"].isin(range(2003,2018))]
tour_train.drop(["DayNum","WScore","LScore","WLoc","NumOT"], axis=1, inplace=True)


# merge seed 
df_winseeds = tour_seed.rename(columns={'TeamID':'WTeamID', 'SeedINT':'WSeed'})
df_lossseeds = tour_seed.rename(columns={'TeamID':'LTeamID', 'SeedINT':'LSeed'})

df_dummy = pd.merge(left=tour_comp_result, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat = df_concat.loc[:,["Season","WTeamID","LTeamID","WSeed","LSeed","SeedDiff"]]

tour_train = tour_train.merge(df_concat, on=["Season","WTeamID","LTeamID"])

stats_train = concat_stats(range(2003,2018))
tour_train = tour_train.merge(stats_train, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
tour_train = tour_train.merge(stats_train, left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], suffixes=("_W", "_L"))
tour_train.drop(["TeamID_W", "TeamID_L"], axis=1, inplace=True)

# calculate features 
for i in ['CoachRegWinRate_W', 
          'CoachTourWinRate_W',
          'FGR_W', 'FGR3_W', 'FTR_W', 'ORR_W',
        'DRR_W', 'AstR_W', 'TOR_W', 'StlR_W', 'BlkR_W', 'PFR_W',
         'CoachRegWinRate_L', 'CoachTourWinRate_L',
         'FGR_L', 'FGR3_L', 'FTR_L', 'ORR_L',
       'DRR_L', 'AstR_L', 'TOR_L', 'StlR_L', 'BlkR_L', 'PFR_L']:
    tour_train[i.split("_")[0]+"Diff"] = tour_train[i] - tour_train[i.split("_")[0]+"_L"]
    if i == "PFR_W":
        break
        
tour_train_win = tour_train.loc[:, ['SeedDiff', 'CoachRegWinRateDiff', 'CoachTourWinRateDiff',
        'FGRDiff', 'FGR3Diff', 'ORRDiff', 'DRRDiff',
       'AstRDiff', 'TORDiff', 'StlRDiff', 'BlkRDiff', 'PFRDiff']]
tour_train_win["Result"] = 1

tour_train_lose = tour_train_win.apply(lambda x: np.where(x == 0, x, -x), axis=0)
tour_train_lose["Result"] = 0

train_df = pd.concat((tour_train_win, tour_train_lose))


"""
After make training dataset 
Let's make our test dataset
"""

def get_season_t1_t2(ID):
    return (int(x) for x in ID.split('_'))

def get_x_test(sub_file):
    """
    sub_file: submission file, either 'sub1' or 'sub2'
    Return: DataFrame for test
    """
    x_test = pd.DataFrame(data=np.zeros(shape=(len(sub_file),4)), 
                          columns=["Season","T1","T2","SeedDiff"], 
                          dtype=int)
    for i, row in sub_file.iterrows():
        season, t1, t2 = get_season_t1_t2(row.ID)
        t1_seed = tour_seed[(tour_seed.TeamID == t1) & (tour_seed.Season == season)].SeedINT.values[0]
        t2_seed = tour_seed[(tour_seed.TeamID == t2) & (tour_seed.Season == season)].SeedINT.values[0]
        diff_seed = t1_seed - t2_seed
        
        x_test.iloc[i,:] = [season, t1, t2, diff_seed]
    
    return x_test


x_test = get_x_test(sub2)
#test_stats = concat_stats(range(2014,2018))
test_stats = concat_stats(range(2018,2019))

x_test = x_test.merge(test_stats, left_on=["Season","T1"], right_on=["Season","TeamID"])
x_test = x_test.merge(test_stats, left_on=["Season",'T2'], right_on=["Season","TeamID"], suffixes=("_S","_B"))

# calculate features 
for i in ['CoachRegWinRate_S', 
          'CoachTourWinRate_S',
          'FGR_S', 'FGR3_S', 'FTR_S', 'ORR_S',
        'DRR_S', 'AstR_S', 'TOR_S', 'StlR_S', 'BlkR_S', 'PFR_S',
         'CoachRegWinRate_B', 'CoachTourWinRate_B',
         'FGR_B', 'FGR3_B', 'FTR_B', 'ORR_B',
       'DRR_B', 'AstR_B', 'TOR_B', 'StlR_B', 'BlkR_B', 'PFR_B']:
    x_test[i.split("_")[0]+"Diff"] = x_test[i] - x_test[i.split("_")[0]+"_B"]
    if i == "PFR_S":
        break
        
test_df = x_test.loc[:, ['SeedDiff', 'CoachRegWinRateDiff', 'CoachTourWinRateDiff',
        'FGRDiff', 'FGR3Diff', 'ORRDiff', 'DRRDiff',
       'AstRDiff', 'TORDiff', 'StlRDiff', 'BlkRDiff', 'PFRDiff']]


"""
Train some models 
"""

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier

x_train = train_df.iloc[:, range(train_df.shape[1] - 1)].values
y_train = train_df.Result.values
x_train, y_train = shuffle(x_train, y_train)

SS = StandardScaler()
x_train = SS.fit(x_train).transform(x_train)

# Logistic classifier with cross-validation 
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=5, num=1000)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True, cv=5)
clf.fit(x_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

# Random Forest classifier with cross-validation 
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
params = {"n_estimators":range(1,10), "max_depth":range(1,10)}
clf = GridSearchCV(rfc, params, scoring="neg_log_loss", refit=True, cv=5)
clf.fit(x_train, y_train)
print('Best log_loss: {:.4}, best n_estimators: {}, max_depth: {}'.\
format(clf.best_score_, clf.best_params_['n_estimators'], clf.best_params_["max_depth"]))

# Try Gradient Boosting method 
import lightgbm as lgb


# Make prediction submit to kaggle
preds = clf.predict_proba(test_df)[:,1]
clipped_preds = np.clip(preds, 0.05, 0.95)
sub2.Pred = clipped_preds

sub2.to_csv("madness.csv", index = False)


