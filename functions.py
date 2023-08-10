import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, cross_val_score

def missing_data_df(df):
    """

    """

    missing_data_count = df.isnull().sum()
    missing_data_percent = df.isnull().sum() / len(df) * 100

    missing_data = pd.DataFrame({'Count': missing_data_count,
                                 'Percent': missing_data_percent})
    
    missing_data = missing_data[missing_data.Count > 0]
    missing_data.sort_values(by='Count', ascending=False, inplace=True)

    return missing_data


def impute_null(df):   
    
    # We have different types of null values:
    # On numerical variables where N/A means no feature we will impute with 0:
    numerical_na = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    df[numerical_na] = df[numerical_na].fillna(0)


    # On categorical variables where N/A means no feature we will impute with 'None':
    categorical_na = ['MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 
                    'GarageType', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
    df[categorical_na] = df[categorical_na].fillna('None')
    

    # The remaining features with null values will be imputed as followed:
    # MasVnrType, MasVnrArea, Electrical, LotFrontage, GarageYrBlt
    df.loc[df.GarageYrBlt.isna(), 'GarageYrBlt'] = df[df.GarageYrBlt.isna()].YearBuilt
    df.loc[df.Electrical.isna(), 'Electrical'] = df.Electrical.mode()[0]
    df.loc[df.MasVnrType.isna(), 'MasVnrType'] = 'None'
    df.loc[df.MasVnrArea.isna(), 'MasVnrArea'] = 0

    df.loc[(df.LotFrontage.isna()) & (df.LotConfig=='CulDSac'), 'LotFrontage'] = 47.5
    df.loc[(df.LotFrontage.isna()) & (df.LotConfig=='FR2'), 'LotFrontage'] = 60
    df.loc[(df.LotFrontage.isna()) & (df.LotConfig=='FR3'), 'LotFrontage'] = 62.5
    df.loc[(df.LotFrontage.isna()) & (df.LotConfig=='Inside'), 'LotFrontage'] = 65
    df.loc[(df.LotFrontage.isna()) & (df.LotConfig=='Corner'), 'LotFrontage'] = 80

    return df

    
def add_location(x):
    if 'SWISU' in x or 'OldTown' in x or 'MeadowV' in x or 'Edwards' in x or 'IDOTRR' in x or 'NPkVill' in x or 'BrkSide' in x:
        return 1
    elif 'NAmes' in x or 'BrDale' in x or 'NWAmes' in x or 'Sawyer' in x or 'Blmngtn' in x or 'Landmrk' in x or 'SawyerW' in x:
        return 2
    elif 'Mitchel' in x or 'ClearCr' in x or 'Blueste' in x or 'CollgCr' in x or 'Crawfor' in x or 'Gilbert' in x or 'StoneBr' in x:
        return 3
    else:
        return 4
    
    

def add_roadrail1(x):
    if 'Artery' in x:
        return 1
    elif 'RRAn' in x:
        return 1
    elif 'RRNn' in x:
        return 1
    elif 'RRAe' in x:
        return 1
    elif 'RRNe' in x:
        return 1
    else:
        return 0

    

def modify_features(df):
    """

    """

    # Merge underpopulated categories
    df.loc[df['GarageCars']>3, 'GarageCars'] = 3
    df.loc[df['TotRmsAbvGrd']>9, 'TotRmsAbvGrd'] = 9
    df.loc[df['Fireplaces']>2, 'Fireplaces'] = 2
    df.loc[df['BsmtQual']=='Po', 'BsmtQual'] = 'None'
    df.loc[df['LotShape']=='IR3', 'LotShape'] = 'IR2'
    #df.loc[df['KitchenQual']=='Po', 'KitchenQual'] = 'Fa'
    df.CentralAir = df.CentralAir.map({'Y': 1, 'N': 0})

    df.loc[df['Foundation']=='Stone', 'Foundation'] = 'BrkTil'
    df.loc[df['Foundation']=='Wood', 'Foundation'] = 'PConc'

    # Create new features
    #df.loc[df['YearBuilt']<1940, 'YearBuilt'] = 1940
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['Location'] = df.Neighborhood.map(add_location)
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'] 
    df['TotalBath'] = df.FullBath + df.BsmtFullBath + 0.5 * (df.HalfBath + df.BsmtHalfBath)
    df['TotalFullBath'] = df.FullBath + df.BsmtFullBath
    df['TotalHalfBath'] = df.HalfBath + df.BsmtHalfBath
    df.loc[df['TotalBath']>4, 'TotalBath'] = 4
    df['RoadRail'] = df.Condition1.map(add_roadrail1)
    df['BedroomPerSF'] = df['BedroomAbvGr']/df['TotalSF']


    # Binarize features 
    df.loc[df['GarageType'].isin(['Attchd', 'BuiltIn']), 'GoodGarageType'] = 1
    df.loc[df['MSZoning'].isin(['RL', 'FV']), 'Zone'] = 1
    df.loc[df['LotConfig'].isin(['CulDSac']), 'CulDSac'] = 1
    df.loc[~df['Exterior1st'].isin(['CemntBd', 'VinylSd']), 'Exterior1st_top'] = 1
    df.loc[df['FlrSF2nd']>0, 'TwoStory'] = 1
    df.loc[df['OverallQual']>7, 'ExQual'] = 1
    df.loc[df['GrLivArea']>=df.GrLivArea.mean(), 'LargerHouse'] = 1
    df.loc[df['YearRemodAdd']>df['YearBuilt'], 'Remod'] = 1
    df.loc[df['BsmtQual']=='Ex', 'ExBsmtQual'] = 1
    df.loc[df['Fireplaces']>0, 'HasFireplace'] = 1
    df.loc[df['KitchenQual']=='Ex', 'ExKitchen'] = 1

    # Fill nulls created by .loc
    df = df.fillna(0)

    return df
    


def encode_ordinal(df):
    """

    """

    ordinal_features = ['ExterQual', 'BsmtQual', 'KitchenQual', 'ExterCond']

    ordinal_encoder = OrdinalEncoder(categories=[['Fa', 'TA', 'Gd', 'Ex'], 
                                                ['None', 'Fa', 'TA', 'Gd', 'Ex'],
                                                ['Po','Fa', 'TA', 'Gd', 'Ex'],
                                                ['Po', 'Fa', 'TA', 'Gd', 'Ex']])

    X = ordinal_encoder.fit_transform(df[ordinal_features])
    X = pd.DataFrame(X, columns=ordinal_features)

    df.drop(columns=ordinal_features, inplace=True)
    df = pd.concat([df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    return df


def dummify_features(df):
    """
    """

    df = pd.get_dummies(df
               ,columns = ['Location']
               ,drop_first = True
               )
    
    df = pd.get_dummies(df
               ,columns = ['Foundation']
               ,drop_first = True
               )
    
    df = pd.get_dummies(df
               ,columns = ['BldgType']
               ,drop_first = True
               )
    
    return df


def df_engineered(df, Reg=True):

    df.rename(columns = {'1stFlrSF':'FlrSF1st', '2ndFlrSF':'FlrSF2nd'}, inplace = True)
    df = df[df.SaleCondition == 'Normal']
    df = df[~df.MSZoning.isin(['C (all)', 'I (all)', 'A (all)', 'A (agr)'])] 

    df = impute_null(df)
    df = modify_features(df)
    df = encode_ordinal(df)

    if Reg:
        df = dummify_features(df)

    return df


def r2rmse_scores(model, X, y):

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    # R^2 
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)

    # RMSE
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))


    mean_r2 = np.mean(r2_scores)
    mean_rmse = np.mean(rmse_scores)

    # Print the scores

    print("-" * 50)
    print("5-fold Cross Validation Scoring")
    print("Mean R^2 score:", mean_r2)
    print("Mean RMSE score:", mean_rmse)
    print("-" * 50)