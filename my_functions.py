import calendar
import numpy as np
import pandas as pd 

def clean_data(df):
    # Drop columns
    df.drop(columns=['id'], axis=1, inplace=True)
    
    # Fill missing values
    df['waterfront'].fillna(df['waterfront'].mode()[0], inplace=True)
    df['view'].fillna(df['view'].mode()[0], inplace=True)
    df['yr_renovated'].fillna(0, inplace=True)
    
    # Replace '?' in sqft_basement with 0.0
    df['sqft_basement'] = df.sqft_basement.replace('?', 0.0)
    
    # Convert sqft_basement to float
    df['sqft_basement']= df.sqft_basement.astype('float64')
  
   # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract month and year from date column
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    
    # Replace appropriate values in 'view' , 'waterfront', 'grade' and 'condition' columns
    df['view'] = df['view'].replace({'NONE': 0, 'AVERAGE': 3, 'GOOD': 4, 'FAIR': 2, 'EXCELLENT': 5})
    df['waterfront'] = df['waterfront'].replace({'NO': 0, 'YES': 1})
    df['grade'] = df['grade'].replace({3: 'Poor', 4: 'Low', 5: 'Fair', 6: 'Low Average', 7: 'Average', 8: 'Good', 9: 'Better', 10: 'Very Good', 11: 'Excellent', 12: 'Luxury', 13: 'Mansion'})
    df['condition'] = df['view'].replace({'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Very Good': 5})
    
    # Convert 'view', 'waterfront', 'grade' and 'condition' columns to apppropriate datatype
    df['view'] = df['view'].fillna(0).astype('int64')
    df['waterfront'] = df['waterfront'].astype('int64')
    df['grade'] = df['view'].fillna(0).astype('int64')
    df['condition'] = df['condition'].astype('int64')

    # Drop original date column
    df.drop(columns=['date'], axis=1, inplace=True)
    
    # Convert year_built to age
    df['age'] = 2015 - df.yr_built
    df = df.drop(columns=['yr_built'], axis=1)
    
    # Fill missing values
    df.yr_renovated.fillna(0.0, inplace=True)
    
    # Create renovated column
    df['renovated'] = df.year - df.yr_renovated
    renovated = df.renovated.values
    age = df.age.values
    values = np.where(renovated <= 10, 1, 0)
    df['renovated'] = np.where(age <= 5, 1, values)

    # Use qcut() function to categorize the 'price' column into three categories
    df['price_categories'] = pd.qcut(df['price'], q=3, labels=['lowest', 'medium', 'highest'])

    # 
    #df['bathrooms'] = np.ceil(df['bathrooms']*2)/2
    #df['bath_categories'] = pd.cut(df['bathrooms'], bins=[0, 1.5, 2.5, df['bathrooms'].max()], labels=['small', 'medium', 'large'])

  
    return df
