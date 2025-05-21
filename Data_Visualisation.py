import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv(r'D:\Python Projects\Learning\Machine Learning\Solving problem\Spendings\Percent_Change_in_Consumer_Spending.csv')
d = data['Date'].drop_duplicates()
codes = data['State FIPS code'].drop_duplicates()
spending_data_columns = ['Accommodation and food service (ACF) spending', 
                         'Arts, entertainment, and recreation (AER)  spending', 
                         'General merchandise stores (GEN) and apparel and accessories (AAP) spending', 
                         'Grocery and food store (GRF)  spending', 
                         'Health care and social assistance (HCS) spending ', 
                         'Transportation and warehousing (TWS)  spending', 
                         'Retail spending, including grocery  (AAP, CEC, GEN, GRF, HIC, ETC, SGH) ',
                          'Retail spending, excluding grocery ((AAP, CEC, GEN, HIC, ETC, SGH) ']
data['total spending'] = data[spending_data_columns].sum(axis =1)
data.to_csv(r'D:\Python Projects\Learning\Machine Learning\Solving problem\Spendings\Percent_Change_in_Consumer_Spending.csv',
                                 index=False)
# print(data.columns.tolist())

len_date = len(d)
len_codes = len(codes)

# print(data.head(10))
# print(d)
# print(len_date)
# print(codes)
# print(data['total spending'].head(10))

# Total spending with respect to date
total_wrt_date = data.groupby('Date')['total spending'].sum()
# print(total_wrt_date)
# plt.figure(figsize=(144,6))
# plt.bar(d , total_wrt_date)
# plt.xlabel('Date')
# plt.ylabel('Total Spending')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Ensure Date is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Create a new column with Month Name and Year
data['Month_Year'] = data['Date'].dt.strftime('%B %Y')

# Initialize dictionary to store month-year -> list of dates
dates_by_month_and_year = {}

# Get unique Month-Year combinations
months_and_year = data['Month_Year'].unique()

# Loop through each month-year and collect the corresponding dates
for month in months_and_year:
    dates_by_month_and_year[month] = data[data['Month_Year'] == month]['Date'].unique().tolist()

# print(dates_by_month_and_year)
total_wrt_month_year = data.groupby('Month_Year')['total spending'].sum()
ACF_wrt_month_year = data.groupby('Month_Year')['Accommodation and food service (ACF) spending'].sum()
AER_wrt_month_year = data.groupby('Month_Year')['Arts, entertainment, and recreation (AER)  spending'].sum()
GEN_wrt_month_year = data.groupby('Month_Year')['General merchandise stores (GEN) and apparel and accessories (AAP) spending'].sum()
GRF_wrt_month_year = data.groupby('Month_Year')['Grocery and food store (GRF)  spending'].sum()
HCS_wrt_month_year = data.groupby('Month_Year')['Health care and social assistance (HCS) spending '].sum()
TWS_wrt_month_year = data.groupby('Month_Year')['Transportation and warehousing (TWS)  spending'].sum()
RSI_wrt_month_year = data.groupby('Month_Year')['Retail spending, including grocery  (AAP, CEC, GEN, GRF, HIC, ETC, SGH) '].sum()
RSE_wrt_month_year = data.groupby('Month_Year')['Retail spending, excluding grocery ((AAP, CEC, GEN, HIC, ETC, SGH) '].sum()
# print(total_wrt_month_year)

plt.figure(figsize=(144,60))
plt.plot(months_and_year,total_wrt_month_year,label = 'Total Spendings')
plt.plot(months_and_year,ACF_wrt_month_year,label = 'Accommodation and food service (ACF) spending')
plt.plot(months_and_year,AER_wrt_month_year,label = 'Arts, entertainment, and recreation (AER)  spending')
plt.plot(months_and_year,GEN_wrt_month_year,label = 'General merchandise stores (GEN) and apparel and accessories (AAP) spending')
plt.plot(months_and_year,GRF_wrt_month_year,label = 'Grocery and food store (GRF)  spending')
plt.plot(months_and_year,HCS_wrt_month_year,label = 'Health care and social assistance (HCS) spending ')
plt.plot(months_and_year,TWS_wrt_month_year,label = 'Transportation and warehousing (TWS)  spending')
plt.plot(months_and_year,RSI_wrt_month_year,label = 'Retail spending, including grocery  (AAP, CEC, GEN, GRF, HIC, ETC, SGH) ')
plt.plot(months_and_year,RSE_wrt_month_year,label = 'Retail spending, excluding grocery ((AAP, CEC, GEN, HIC, ETC, SGH) ')
plt.legend()
plt.xlabel('Month and Year')
plt.ylabel('Total Spending')
plt.xticks(rotation=45)

# Add horizontal reference line at average spending
avg_spending = sum(total_wrt_month_year) / len(total_wrt_month_year)
plt.axhline(y=avg_spending, color='red', linestyle='--', label=f'Average: {avg_spending:.2f}')

for i, value in enumerate(total_wrt_month_year):
    plt.annotate(f'{value:.1f}',            # The value to display
                 (i, value),                # Coordinates of the point
                 textcoords="offset points",
                 xytext=(0, 8),             # Offset upward
                 ha='center', fontsize=9)   # Center align text

plt.tight_layout()
plt.show()

total_wrt_code = data.groupby('State FIPS code')['total spending'].sum()
plt.figure(figsize=(12,6))
plt.plot(codes,total_wrt_code, marker = 'o', label = 'Total Spendings')
plt.xlabel('State FIPS code')
plt.ylabel('Total Spending')
plt.xticks(rotation=45)

for i, value in enumerate(total_wrt_code):
    plt.annotate(f'({codes[i]},{value:.1f})',            # The value to display
                 (codes[i], value),                # Coordinates of the point
                 textcoords="offset points",
                 xytext=(0, 8),             # Offset upward
                 ha='center', fontsize=9)   # Center align text
    
plt.show()

