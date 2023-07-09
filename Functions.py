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
    # On numerical variables where N/A means no feature we will impute with 'None':
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
    df.loc[df.LotFrontage.isna(), 'LotFrontage'] = df.LotFrontage.mean()
    df.loc[df.Electrical.isna(), 'Electrical'] = df.Electrical.mode()[0]
    df.loc[df.MasVnrType.isna(), 'MasVnrType'] = 'None'
    df.loc[df.MasVnrArea.isna(), 'MasVnrArea'] = 0

    return df


def modify_features(df):
    """

    """

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    df.loc[df['GarageCars']>3, 'GarageCars'] = 3
    df.loc[df['TotRmsAbvGrd']>9, 'TotRmsAbvGrd'] = 9
    df.loc[df['Fireplaces']>2, 'Fireplaces'] = 2
    df.loc[df['BsmtQual']=='Po', 'BsmtQual'] = 'None'

    df.loc[df['GarageType'].isin(['Attchd', 'BuiltIn']), 'GoodGarageType'] = 1

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