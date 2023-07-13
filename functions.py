import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

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
    if 'MeadowV' in x or 'BrDale' in x or 'IDOTRR' in x or 'OldTown' in x or 'Blueste' in x or 'Edwards' in x or 'BrkSide' in x:
        return 1
    elif 'Sawyer' in x or 'Landmrk' in x or 'SWISU' in x or 'NAmes' in x or 'NPkVill' in x or 'Mitchel' in x or 'NWAmes' in x:
        return 2
    elif 'Gilbert' in x or 'SawyerW' in x or 'Blmngtn' in x or 'Crawfor' in x or 'CollgCr' in x or 'ClearCr' in x or 'Greens' in x:
        return 3
    else:
        return 4
    

def modify_features(df):
    """

    """

    # Merge uderpopulated categories
    df.loc[df['GarageCars']>3, 'GarageCars'] = 3
    df.loc[df['TotRmsAbvGrd']>9, 'TotRmsAbvGrd'] = 9
    df.loc[df['Fireplaces']>2, 'Fireplaces'] = 2
    df.loc[df['BsmtQual']=='Po', 'BsmtQual'] = 'None'
    df.loc[df['LotShape']=='IR3', 'LotShape'] = 'IR2'

    df.loc[df['Foundation']=='Stone', 'Foundation'] = 'BrkTil'
    df.loc[df['Foundation']=='Wood', 'Foundation'] = 'PConc'

    # Create new features
    df.loc[df['YearBuilt']<1940, 'YearBuilt'] = 1940
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['Location'] = df.Neighborhood.map(add_location)
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'] 


    # Binarize features where only a few categories appear to be correlated to SalesPrice.
    df.loc[df['GarageType'].isin(['Attchd', 'BuiltIn']), 'GoodGarageType'] = 1
    df.loc[df['MSZoning'].isin(['RL', 'FV']), 'Zone'] = 1
    df.loc[df['LotConfig'].isin(['CulDSac']), 'CulDSac'] = 1
    df.loc[~df['Exterior1st'].isin(['CemntBd', 'VinylSd']), 'Exterior1st_top'] = 1
    df.loc[df['FlrSF2nd']>0, 'TwoStory'] = 1
    df.loc[df['OverallQual']>7, 'ExQual'] = 1
    df.loc[df['GrLivArea']>=df.GrLivArea.mean(), 'LargerHouse'] = 1
    df.loc[df['YearRemodAdd']>df['YearBuilt'], 'Remod'] = 1
    df.loc[df['BsmtQual']=='Ex', 'ExBsmtQual'] = 1
    df.loc[df['YearRemodAdd']>df['YearBuilt'], 'Remod'] = 1

    # Fill nulls created by .loc
    df = df.fillna(0)

    return df
    


def encode_ordinal(df):
    """

    """

    ordinal_features = ['ExterQual', 'BsmtQual', 'KitchenQual', 'ExterCond']

    ordinal_encoder = OrdinalEncoder(categories=[['Fa', 'TA', 'Gd', 'Ex'], 
                                                ['None', 'Fa', 'TA', 'Gd', 'Ex'],
                                                ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
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


def df_engineered(df):

    df = df[df.SaleCondition == 'Normal']
    df = df[~df.MSZoning.isin(['C (all)', 'I (all)', 'A (all)', 'A (agr)'])] 

    df = impute_null(df)
    df = modify_features(df)
    df = encode_ordinal(df)
    df = dummify_features(df)

    # Training and test sets 
    #df_2010 = df[df['YrSold']==2010].reset_index(drop=True)
    #df = df[df['YrSold']<2010].reset_index(drop=True)

    #return {'train':df, 'test':df_2010}
    return df