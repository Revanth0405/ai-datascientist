import pandas as pd
from sklearn.impute import SimpleImputer

def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    for col in df.select_dtypes(include='object'):
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    num_cols = df.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df