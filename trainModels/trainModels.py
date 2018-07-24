import numpy as np 
import pandas as pd 

# load interested dataset 
# input some data of interest 
path = '~/NCAA'

tour_seed = pd.read_csv(path + "/NCAATourneySeeds.csv") # 1985-2018
tour_comp_result = pd.read_csv(path + "/NCAATourneyCompactResults.csv") # 1985-2017
reg_detailed_result = pd.read_csv(path + "/RegularSeasonDetailedResults.csv") # 2003-2018
reg_comp_result = pd.read_csv(path + "/RegularSeasonCompactResults.csv") # 1985-2018
#sub1 = pd.read_csv(path + "/SampleSubmissionStage1.csv") # 2014-2017
sub2 = pd.read_csv(path + "/SampleSubmissionStage2.csv") # 2018
coaches = pd.read_csv(path + "/TeamCoaches.csv") # 1985-2018


"""
This part build couple of convenient functions to get:
	+ coach life-time regular season winning rate 
	+ coach life-time tournament winning rate 
	+ team regular season detailed statistics, e.g, rebound per game, block per game, etc.
	+ make train and submission dataset 
"""

# calculate coach regular season winning rate
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
    
    tour_coaches = pd.merge(tour_seed[tour_seed.Season==season],
                            coaches[coaches["LastDayNum"]==154],
                            left_on=["Season", "TeamID"], 
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
    
  
    tour_coaches = pd.merge(tour_seed[tour_seed.Season==season],
                            coaches[coaches["LastDayNum"]==154], 
                            left_on=["Season", "TeamID"], 
                            right_on=["Season", "TeamID"])
    
    tour_coaches_win_rate = tour_concat.merge(tour_coaches,how="right", on="CoachName")
   
    # when coaches has not tournament data, fill winning rate with 0.5 as non-informative guess
    tour_coaches_win_rate.fillna(value=0.5, inplace=True) 
    return tour_coaches_win_rate.loc[:, ["Season", "TeamID", "CoachTourWinRate"]]

#select columns (detailed statistics) for each team
#W and L stand for winning and losing team
#FGA: field goal attempt 
#FGM3: 3-points made
#FGA3: 3-points attempt
#FTM: free throw made
#FTA: free throw attempt
#OR: ofense rebound 
#DR: defense rebound
#Ast: assistant 
#TO: turn over
#Stl: steal
#Blk: block
#PF: personal foul

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

#function collecting regular season statistics 
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

#function concating statistics for all seasons
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


# extract season, team id from string 
def get_season_t1_t2(ID):
    return (int(x) for x in ID.split('_'))
    

x_test = pd.DataFrame(data = np.zeros(shape = (len(sub2), 4)),
                     columns = ["Season", "T1", "T2", "SeedDiff"],
                     dtype = int)

def get_x_test(sub_file):
    """
    sub_file: submission file, either 'sub1' or 'sub2'
    Return: DataFrame for test
    """
    for row in sub_file.itertuples():
        season, t1, t2 = get_season_t1_t2(row.ID)
        t1_seed = tour_seed[(tour_seed.TeamID == t1) & (tour_seed.Season == season)].SeedINT.values[0]
        t2_seed = tour_seed[(tour_seed.TeamID == t2) & (tour_seed.Season == season)].SeedINT.values[0]
        diff_seed = t1_seed - t2_seed
        
        x_test.iloc[row.Index, :] = [season, t1, t2, diff_seed]
    
    return x_test

# make test dataset 
x_test = get_x_test(sub2)
test_stats = concat_stats(range(2018,2019))

x_test = x_test.merge(test_stats,
                      left_on = ["Season", "T1"],
                      right_on = ["Season", "TeamID"])

x_test = x_test.merge(test_stats,
                      left_on=["Season",'T2'],
                      right_on=["Season","TeamID"],
                      suffixes=("_S","_B"))

# calculate difference in each statistics 
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
Below is training process, three models were trained as candidates.
* Logistic regresion:
	+ Normalize feature 
	+ Grid search over regularization parameter and type of regularization 
	+ Fine tune after first round of tunning 
* Random forest classifier: 
	+ Tuning parameters are: 
		number of trees, 
		judging criterion, 
		number of features, 
		tree depth, 
		minimum samples needed for each split 
	+ number of trees and max tree depth come first
	+ Use best parameters above, tune the rest ones 
	+ Fine tune after first round 
* Light GBM classifier:
	+ Tunning parameters are: 
		number of leaves for each tree,
		max tree depth,
		number of boost round,
		minimum samples needed for each leave,
		minimum child weight,
		number of samples for each boosting round,
		number of features for each boosting round,
		learning rate
	+ First find best number of trees, fix number of boosting round 
	+ Tune max depth, number of leaves 
	+ Tune minimum samples needed for each leave, minimum child weight
	+ Tune row/col samples proportion needed for random sample at each boosting round 
	+ Tune learning rate 
	+ Fine tune after first round 
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb 
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# normalize features for logistic regression 
predictors = ['SeedDiff', 'CoachRegWinRateDiff', 
              'CoachTourWinRateDiff', 'FGRDiff',
              'FGR3Diff', 'ORRDiff', 'DRRDiff', 
              'AstRDiff', 'TORDiff', 'StlRDiff',
              'BlkRDiff', 'PFRDiff']
target = ['Result']

x_train = train_df.loc[:, predictors].values
y_train = train_df.Result.values

SS = StandardScaler()
x_train = SS.fit(x_train).transform(x_train)

# Try Logistic regression as base line model 
LR = LogisticRegression(random_state = 888)

param = {'penalty': ['l2'],
        'C': [0.031]}

LRGrid = GridSearchCV(estimator = LR,
                     param_grid = param,
                     scoring = 'neg_log_loss',
                     n_jobs = 6,
                     cv = 5,
                     verbose = 1
                     )

LRGrid.fit(x_train, y_train)

print("best params: {}.\nbest score: {}".format(LRGrid.best_params_, LRGrid.best_score_))

# try random forest

x_train = train_df.loc[:, predictors].values
y_train = train_df.Result.values

rfc = RandomForestClassifier(random_state = 888)

param = {'n_estimators': [26],
        'criterion': ['entropy'],
        'max_features': [8],
        'max_depth': [5],
        'min_samples_split': [12]}

rfcGrid = GridSearchCV(estimator = rfc,
                      param_grid = param,
                      scoring = 'neg_log_loss',
                      n_jobs = 6,
                      cv = 5,
                      verbose = 1)

rfcGrid.fit(x_train, y_train)

print("best params: {}.\nbest score: {}".format(rfcGrid.best_params_, rfcGrid.best_score_))


# try lightgbm classifier 

x_train = train_df.loc[:, predictors].values
y_train = train_df.Result.values

param = {
    'num_leaves': [12], 
    'max_depth': [2],
    'min_child_samples': [28],
    'n_estimators': [25],
    'min_child_weight': [1.5],
    'subsample': [1],
    'colsample_bytree': [0.5],
    'learning_rate': [0.25]
    }

if __name__ == '__main__':
    lgbGrid = GridSearchCV(
        estimator = LGBMClassifier(
            boosting_type = 'gbdt',
            objective = 'binary',
            is_unbalance = False,
            min_split_gain = 0,
            random_state = 888,
            num_thread = 6
            ),
        param_grid = param,
        scoring = 'neg_log_loss',
        n_jobs = 6,
        cv = 5,
        verbose = 1
        )
    
lgbGrid.fit(x_train, y_train)

print("best params: {}\nbest score: {}".format(lgbGrid.best_params_, lgbGrid.best_score_))


# create submission file 

LRPred = LRGrid.predict_proba(test_df)[:, 1]
rfcPred = rfcGrid.predict_proba(test_df)[:, 1]
lgbPred = lgbGrid.predict_proba(test_df)[:, 1]

# avoid infinit penalty for being too confident on wrong prediction 
pred_blend = np.mean([LRPred, rfcPred, lgbPred], axis = 0)
clipped_pred = np.clip(pred_blend, 0.05, 0.95)

sub2['Pred'] = clipped_pred 

sub2.to_csv('blend_sub.csv', index = False)
