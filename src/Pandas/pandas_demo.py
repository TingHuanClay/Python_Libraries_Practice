import pandas as pd

groceries = None


def pands_series_demo():
    global groceries
    groceries = pd.Series(data=[30, 6, 'Yes', 'No'], index=[
        'eggs', 'apple', 'milk', 'bread'])
    print(groceries)
    """
    Output:
        eggs      30
        apple      6
        milk     Tes
        bread     No
        dtype: object
    """

    print(f"groceries.shape: {groceries.shape}")
    print(f"groceries.ndim: {groceries.ndim}")
    print(f"groceries.size: {groceries.size}")
    """
    Output:
        groceries.shape: (4,)
        groceries.ndim: 1
        groceries.size: 4
    """

    print(f"groceries.index: {groceries.index}")
    print(f"groceries.values: {groceries.values}")
    """
    Output:
        groceries.index: Index(
            ['eggs', 'apple', 'milk', 'bread'], dtype='object')
        groceries.values: [30 6 'Tes' 'No']
    """

    # Check is there specific item in the index sets
    is_contain_banana = 'banana' in groceries
    print(f"is_contain_banana: {is_contain_banana}")
    is_contain_bread = 'bread' in groceries
    print(f"is_contain_bread: {is_contain_bread}")


def pands_series_access_demo():
    print('\n============= pands_series_access_demo [START] ============= ')
    print('\nUsing the value of Index to get the elements in the series')
    print(f"groceries['eggs'] : {groceries['eggs']}")
    # Double Quote
    print(f"groceries['milk', 'bread'] : \n{groceries[['milk', 'bread']]}")

    # Using Index to get the first and the last element in the series
    print('\nUsing Index to get the first and the last element in the series')
    print(f"groceries[0] : {groceries[0]}")
    print(f"groceries[-1] : {groceries[-1]}")
    print(f"groceries[[0, 2]] : {groceries[[0, 2]]}")

    # modify the value by accessing index
    print()
    groceries['eggs'] = 2
    print(groceries)

    # use drop to delete item from the series "OUT of the place"
    print('After dropping Apple:')
    print(groceries.drop('apple'))

    # use drop to delete item from the series "In place" with the paramter
    groceries.drop('apple', inplace=True)
    print(groceries)


def pands_series_operation_demo():
    print('\n\n==== pands_series_operation_demo [START] ==== ')
    fruits = pd.Series([10, 6, 3], ['apples', 'oranges', 'bananas'])
    print(fruits)
    print(fruits + 2)
    print(fruits - 2)
    print(fruits / 2)

    print()
    import numpy as np
    print(f"np.sqrt(fruits):\n{np.sqrt(fruits)}")
    print(f"np.exp(fruits):\n{np.exp(fruits)}")
    print(f"np.power(fruits, 2):\n{np.power(fruits, 2)}")

    # modify element by assign index and value all 'OUT of SPACE'
    print("\nmodify element by assign index and value:\n")
    print(f"fruits['bananas'] + 2: \n{fruits['bananas'] + 2}")
    print(
        f"fruits[['apples', 'oranges']]] * 2: \n{fruits[['apples', 'oranges']] * 2}")
    print(f"fruits.iloc[0] - 2: \n{fruits.iloc[0] - 2}")
    print(
        f"fruits.loc[['apples', 'oranges']] / 2: \n{fruits.loc[['apples', 'oranges']] / 2}")


def pandas_frame_demo():
    # Shopping cart for 2 people with the item value and item names
    items = {
        'Bob': pd.Series([245, 25, 55], index=['bike', 'pants', 'watch']),
        'Alice': pd.Series([40, 110, 500, 45], index=['book', 'glasses', 'bike', 'pants'])
    }
    print(type(items))

    # Convert dictionary to Data Frame
    shopping_carts = pd.DataFrame(items)
    print(shopping_carts)

    # Create dataFrame with 'Default' Index, which is 0, 1, 2, 3...
    data = {
        'Bob': pd.Series([245, 25, 55]),
        'Alice': pd.Series([40, 110, 500, 45])
    }
    df = pd.DataFrame(data)
    print(df)

    # get the attribute of the data frame
    print(f"shopping_carts.values: \n{shopping_carts.values}")
    print(f"shopping_carts.shape: {shopping_carts.shape}")
    print(f"shopping_carts.ndim: {shopping_carts.ndim}")
    print(f"shopping_carts.size: {shopping_carts.size}")

    # Get the subsets from the data frame by COLUMN
    print('Get the subsets from the data frame by "COLUMN"')
    bob_shopping_carts = pd.DataFrame(items, columns=['Bob'])
    print(bob_shopping_carts)

    # Get the subsets from the data frame by INDEX
    print('Get the subsets from the data frame by "INDEX"')
    sel_shopping_carts = pd.DataFrame(items, index=['pants', 'watch'])
    print(sel_shopping_carts)

    # Get the subsets from the data frame by COLUMN & INDEX
    print('Get the subsets from the data frame by "INDEX" & "COLUMN"')
    alice_sel_shopping_carts = pd.DataFrame(
        items, index=['glasses', 'bike'], columns=['Alice'])
    print(alice_sel_shopping_carts)

    # Create Index for the dataframe, even though there is NO index in the 'Dictionary'
    # NO index in the following dictionary
    data = {
        'Integers': [1, 2, 3],
        'Float': [4.5, 8.2, 9.6]
    }
    df = pd.DataFrame(data, index=['label 1', 'label 2', 'label 3'])
    print(df)
    """
        Output:
                    Integers  Float
            label 1         1    4.5
            label 2         2    8.2
            label 3         3    9.6
    """

    # Create Index for the dataframe, even though there is NO index in the 'Array'
    # NO index in the following Array
    items = [
        {'bike': 20, 'pants': 30, 'watches': 35},
        {'watches': 10, 'glasses': 50, 'bike': 15, 'pants': 5}
    ]
    store_item = pd.DataFrame(items, index=['store 1', 'store 2'])
    print(store_item)
    """
        Output:
                    bike  glasses  pants  watches
            store 1    20      NaN     30       35
            store 2    15     50.0      5       10
    """


def pandas_frame_access_demo():
    items = [
        {'bike': 20, 'pants': 30, 'watches': 35},
        {'watches': 10, 'glasses': 50, 'bike': 15, 'pants': 5}
    ]
    store_items = pd.DataFrame(items, index=['store 1', 'store 2'])
    print(store_items)
    print()

    print(store_items[['bike']])
    print()
    print(store_items[['bike', 'pants']])
    print()
    print(store_items.loc[['store 1']])
    print()

    #
    # Please be NOTICED that column label first, then row label comes in Data Frame
    #
    print(store_items['bike']['store 2'])
    print()

    # Add a new item into data frame
    store_items['shirts'] = [15, 2]
    print(store_items)
    print()

    """
        combined existed data in the data frame
    """
    print('\ncombined existed data in the data frame\n')
    # Add a new 'COLUMN' combined from existed columns in the data frame
    store_items['suits'] = store_items['shirts'] + store_items['pants']
    print(store_items)
    print()

    # Add a new 'ROW' combined from existed rows in the data frame
    # You have to create a new data frame and append it to existed data frame
    new_items = [{'bike': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]
    new_store = pd.DataFrame(new_items, index=['store 3'])
    print(new_store)

    store_items = store_items.append(new_store)
    print(store_items)

    # Add a new column from existed column, but only part of rows
    store_items['new_watches'] = store_items['watches'][1:]
    print(store_items)

    # Insert a new column with specified position
    # df.insert(index of column, column name, data)
    store_items.insert(5, 'shoes', [8, 5, 0])
    print(store_items)

    """
        using 'pop()' to delete column
    """
    store_items.pop('new_watches')
    print(store_items)

    """
        using 'drop()' to delete column or row (it depends on the axis specified)
    """
    # Drop column with axis = 1
    store_items = store_items.drop(['watches', 'shoes'], axis=1)
    print(store_items)

    # Drop column with axis = 0
    store_items = store_items.drop(['store 1', 'store 2'], axis=0)
    print(store_items)

    """
    Rename the column in data frame: 'pd.rename(columns: {A: a}, {B: b})'
    """
    store_items = store_items.rename(columns={'bike': 'hats'})
    print(store_items)

    """
    Rename the row in data frame: 'pd.rename(index: {A: a}, {B: b})'
    """
    store_items = store_items.rename(index={'store 3': 'last store'})
    print(store_items)


def pandas_data_clean_demo():
    """
    How to deal with the Nan value in the data frame
             bike  glasses  pants  shirts  shoes  suits  watches
    store 1    20      NaN     30    15.0      8   45.0       35
    store 2    15     50.0      5     2.0      5    7.0       10
    store 3    20      4.0     30     NaN     10    NaN       35
    """
    items = [
        {'bike': 20, 'pants': 30, 'watches': 35,
            'shirts': 15, 'shoes': 8, 'suits': 45},
        {'watches': 10, 'glasses': 50, 'bike': 15,
            'pants': 5, 'shirts': 2, 'shoes': 5, 'suits': 7},
        {'bike': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes': 10}
    ]
    store_items = pd.DataFrame(items, index=['store 1', 'store 2', 'store 3'])
    print(store_items)

    # Count the number of Nan value in the Data frame
    x = store_items.isnull().sum().sum()
    print(f"NaN values count: {x}")

    # Count the number of Not-Nan value in the Data frame
    x = store_items.count().sum().sum()
    print(f"Not-NaN values count: {x}")

    # Drop the ROWS which has NaN value OUT of place
    print(store_items.dropna(axis=0))

    # Drop the COLUMNS which has NaN value "OUT of place"
    print(store_items.dropna(axis=1))

    # Drop the COLUMNS which has NaN value "In place"
    # store_items.dropna(axis=1, inplace=True)

    # fill the Nan value with specified value
    print(store_items.fillna(0))
    # fill the Nan value with PREVIOUS 'row' value
    print(store_items.fillna(method="ffill", axis=0))
    # fill the Nan value with PREVIOUS 'column' value
    print(store_items.fillna(method="ffill", axis=1))
    # fill the Nan value with NEXT 'row' value
    print(store_items.fillna(method="backfill", axis=0))
    # fill the Nan value with NEXT 'column' value
    print(store_items.fillna(method="backfill", axis=1))

    # fill the Nan value with  'column' value
    print(store_items.fillna(method="linear", axis=0))
    print(store_items.fillna(method="linear", axis=1))


def pandas_file_and_groupby_demo():
    google_stock = pd.read_csv('./goog-1.csv')
    print(type(google_stock))
    print(google_stock.shape)
    # print(google_stock)
    print(google_stock.head())
    print(google_stock.head(2))
    print(google_stock.tail())
    print(google_stock.tail(3))

    # Check the Nan value in the Data frame
    print('Check the Nan value in the Data frame')
    print(google_stock.isnull().any())

    # get statistic information of the data frame
    print('get statistic information of the data frame')
    print(google_stock.describe())
    print('get statistic information of the column in the data frame')
    print(google_stock['Adj Close'].describe())
    print('get max statistic information of the data frame')
    print(google_stock.max())
    print('get mean statistic information of the data frame')
    print(google_stock.mean())
    print('get min statistic information of the column in the data frame')
    print(google_stock['Close'].min())

    # Correlation value
    print('\nget correlation information of the data frame')
    print(google_stock.corr())

    # groupby() for aggregation information
    # data.groupby(['Year'])['Salary'].sum()
    # data.groupby(['Year'])['Salary'].mean()
    # data.groupby(['Name'])['Salary'].sum()
    # data.groupby(['Year', 'Department'])['Salary'].sum()


if __name__ == '__main__':
    # pands_series_demo()

    # pands_series_access_demo()

    # pands_series_operation_demo()

    # pandas_frame_demo()

    # pandas_frame_access_demo()

    # pandas_data_clean_demo()

    pandas_file_and_groupby_demo()
