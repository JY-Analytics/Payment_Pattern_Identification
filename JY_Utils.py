import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import plotly
import plotly.graph_objs as go
import networkx as nx
import math


# Helper function to map the day-of-month (1st-31st) to a circle with x/y coordinates
def map_values_to_circle(df):
    min_val = 1 #df.min()
    max_val = 31 #df.max()
    
    # Number of unique values in the range
    value_range = max_val - min_val
    
    # Normalize values to range [0, 2*pi]
    normalized_df = (df - min_val) / value_range * 2 * np.pi
    
    # Calculate x and y coordinates
    x_coords = np.cos(normalized_df)
    y_coords = np.sin(normalized_df)
    
    return x_coords, y_coords

# Helper function to assess wether 2 points on a circle are within the range of a fixed value epsilon
# This is used in the time-proximity critera for the graph algorithim.
def are_points_within_epsilon(x1, y1, x2, y2, epsilon):
    # Calculate the Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Check if the distance is within epsilon
    return distance <= epsilon


# Helper function to filter the dataframe according to various criteria
def filter_transactions(df, 
                        public_token, 
                        min_exp_threshold, 
                        max_exp_threshold, 
                        min_inc_threshold,
                        max_inc_threshold,
                        start_date,
                        end_date):


    # applicant filters
    if (type(public_token) is str) or (type(public_token) is np.str_):
        df_filtered = df[df['public_token']==public_token]
    elif type(public_token) is list:
        df_filtered = df[df['public_token'].isin(public_token)]

    # date filters
    df_filtered = df_filtered[(df_filtered['date_posted']>=start_date) & (df_filtered['date_posted']<=end_date)]

    # income and expense filters
    df_filtered = df_filtered[((df_filtered['amount']<=-min_exp_threshold) & (df_filtered['amount']>=-max_exp_threshold)) | ((df_filtered['amount']>=min_inc_threshold) & (df_filtered['amount']<=max_inc_threshold))]

    return df_filtered




# Helper Function to create the Description-match Matrix 'D'
def create_desc_match_matrix(series, match_method='exact'):
    
    # Get the length of the series
    n = len(series)
    
    # Create an empty n x n matrix
    M = np.zeros((n, n))
    
    if match_method=='exact':
        # Populate the matrix based on matching values
        for i in range(n):
            for j in range(n):
                if (series.iloc[i] == series.iloc[j]):
                    M[i, j] = 1

    elif match_method=='any':
        M = np.ones((n, n))
    
    return M

# Helper Function to create the Amount-match Matrix 'A'
def create_amt_match_matrix(series, dollar_threshold=0):

    
    # Get the length of the series
    n = len(series)
    
    # Create an empty n x n matrix
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            upper_bound = series.iloc[i] + dollar_threshold
            lower_bound = series.iloc[i] - dollar_threshold
            if ((upper_bound > series.iloc[j]) and (lower_bound < series.iloc[j])):
                M[i, j] = 1

    return M


# Helper Function to create the Time-match Matrix 'T'
def create_time_match_matrix(df, month_range=2, day_range=10):

    # Get the length of the series
    n = len(df)
    
    # Create an empty n x n matrix
    M = np.zeros((n, n))

    # Calculate epsilon parameter to test if transaction days are nearby
    C = 2*math.pi*1 #circumference of circle with radius 1, corresponding to xcoord and ycoord mapping
    arc = day_range/31  # arc length
    epsilon = (C/(math.pi))*(np.sin(arc*math.pi/C))

    for i in range(n):
        for j in range(n):
            upper_bound = df['months_since_start'].iloc[i] + month_range
            lower_bound = df['months_since_start'].iloc[i] - month_range
            # first test if the point is within the month range
            if ((upper_bound > df['months_since_start'].iloc[j]) and (lower_bound < df['months_since_start'].iloc[j])):

                # if so, test wether it is within the day range
                x_i = df['xcoord'].iloc[i]
                y_i = df['ycoord'].iloc[i]
                x_j = df['xcoord'].iloc[j]
                y_j = df['ycoord'].iloc[j]
                
                if are_points_within_epsilon(x_i, y_i, x_j, y_j, epsilon):
                    M[i, j] = 1



    
    return M




def recurring_streams(df, 
                      min_recurrences=3, 
                      methods={'desc_match_params': ['any'], 
                               'amt_match_params': [0], 
                               'date_match_params': [2]}):
        
    """
        The main algorithim to identify and group related recurring payment streams.
        

        Args:
            df: The pre-processed dataframe of transactions.
                This should include all the necessary feature engineering

            min_recurrences: The minimum number of recurring payments necessary to be considered part of a stream.
                             For example, if min_recurrences=4 then any streams of <4 transactions will be grouped 
                                 together in the 'all other transactions' category with value of -1. 
            
            methods: A dictionary of lists. Each key refers to the t
                    
                        methods['desc_match_params'][0] can be 'exact' if only identical descriptions will match or 'any' if description is ignored.
                        
                        methods['amt_match_params'][0] represents the dollar threshold needed for transactions to match. For example if 20 is used,
                            then a $200 transaction can match a $180 through $220 transaction. Defaut is zero, meaning only exact $ amount matches will match.

                        methods['time_match_params'][0] is the month range. 
            

        Return:
            payment_stream_group: A pandas series of integers identifying which payment stream the transaction belongs to.
                                  The series is the same length as the input dataframe.
                                  Unassigned transactions will be grouped together with a value of -1.
            
        """
    desc_match_method = methods['desc_match_params'][0]
    dollar_threshold = methods['amt_match_params'][0]
    month_range = methods['time_match_params'][0]
    day_range = methods['time_match_params'][1]
    
    ### Step 1: Create Pairwise Matricies:
    
    # D is a matrix of description-matching
    #   D_ij = 1 if the description of transaction i is an exact match of transaction j's description. D_ij=0 otherwise. 
    D = create_desc_match_matrix(pd.Series(df['description']),
                                    match_method=desc_match_method
                                ).astype(bool)

    # A is a matrix of amount-matching
    #   A_ij = 1 if the transaction amount i is a near-match of transaction j's amount. A_ij=0 otherwise. 
    A = create_amt_match_matrix(pd.Series(df['amount']), 
                                dollar_threshold=dollar_threshold
                                ).astype(bool)

    # T is a matrix of time-matching
    #   T_ij = 1 if the transaction amount i is a near-match of transaction j's amount. A_ij=0 otherwise. 
    T = create_time_match_matrix(df,
                                 month_range=month_range,
                                 day_range=day_range
                                ).astype(bool)

    
    
    ### Step 2: Use the pairwise matricies above to build the adjacency matrix according to the method specified.

    adj_matrix = (D & A) & T # Matrix where all matching conditions are met
    np.fill_diagonal(adj_matrix, 0) # No self-connected nodes


    
    ### Step 3: Build the Graph and unpack the connected components
    
    # build graph from adj_matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Collect recurring payment streams from connected components
    recurring_streams = [c for c in sorted(nx.connected_components(G), key=len, reverse=True) if len(c)>=min_recurrences]

    # All communities will start out as unassigned and given an index of -1
    payment_stream_group = [-1]*nx.number_of_nodes(G)
    
    # Go through each item in each set and assign it to the connected component
    set_ind = 0
    for set in recurring_streams:
        for item in set:
            # If the item was previously unassigned, its assigned to the set its found in
            if payment_stream_group[item] == -1:
                payment_stream_group[item] = set_ind
            # Note: The else condition below should never trigger, since connected components should be mutually exclusive
            # If the item was already in another set, its assigned to -2, which contains multi-community nodes
            else:
                payment_stream_group[item] = -2
                    
        set_ind += 1


    return payment_stream_group

# The main algorithim used to identify late and missed payments
def label_missed_payments(df, groupings, days_late):

    return_df = df.copy()
    # initialize the return columns
    return_df['nearest'] = pd.to_datetime('1999-01-01')
    return_df['days_late'] = 1000
    return_df['missed_pmt'] = False
    
    
    for group in groupings.unique():
        filter = (groupings == group)
        df_temp = return_df[filter]

        indices_to_update = groupings == group
        #values_to_set = grp[grp.str[-36:] == applicant]
        return_df.loc[indices_to_update, 'nearest'] = nearest_dates(pmt_dates=df_temp['date_posted'],
                                                                    target_dates=get_target_dates(df_temp['date_posted'])
                                                                   )#.values
        

    return_df['days_late'] = return_df['date_posted'] - return_df['nearest']

    return_df['late_payment'] = return_df['days_late'].dt.days>days_late


    # test for missed payments
    missed_pmt_df = pd.DataFrame()
    
    for group in groupings.unique():
        filter = (groupings == group)
        df_temp = return_df[filter]

        for expected_pmt in get_target_dates(df_temp['date_posted']):
            if expected_pmt not in [i for i in df_temp['nearest']]:
                
                # Define the fields to copy from the first row
                fields_to_copy = ['public_token', 'group']

                first_row = df_temp.iloc[0]
                new_row = {field: first_row[field] for field in fields_to_copy}
                #new_row = {field: first_row[field] if field in fields_to_copy else np.nan for field in df_temp.columns}
                new_row_df = pd.DataFrame([new_row])
                
                new_row_df['date_posted'] = expected_pmt
                new_row_df['nearest'] = expected_pmt
                new_row_df['missed_pmt'] = True
                
                # Append the new row to the DataFrame
                missed_pmt_df = pd.concat([missed_pmt_df, new_row_df], ignore_index=True)

            
    return_df = pd.concat([return_df, missed_pmt_df], ignore_index=True)

    
    return return_df


def get_target_dates(dates):
    # Find the earliest and latest month in the date range
    start_date = dates.min() - pd.DateOffset(months=1)
    end_date = dates.max() + pd.DateOffset(months=1)
    
    # Generate all months in the range
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Create a series with the median_day for each month
    median_day = np.median(dates.dt.day)
    median_day_series = all_months + pd.DateOffset(days=median_day - 1)

    
    # Ensure that the day does not exceed the number of days in the month
    #median_day_series = median_day_series[median_day_series.dt.day == median_day]
    
    return pd.Series(median_day_series)


def nearest_dates(pmt_dates, target_dates):

    # Convert dates to ordinal numbers for easier comparison
    pmt_ordinals = pmt_dates.map(pd.Timestamp.toordinal).values
    target_ordinals = target_dates.map(pd.Timestamp.toordinal).values
    
    # Find the nearest date for each date in pmt_dates
    nearest_dates = []
    for pmt_date in pmt_ordinals:
        # Compute the absolute differences with target_dates
        differences = np.abs(target_ordinals - pmt_date)
        # Find the index of the minimum difference
        closest_index = differences.argmin()
        # Append the corresponding date from target_dates
        nearest_dates.append(target_dates.iloc[closest_index])
    
    return nearest_dates
