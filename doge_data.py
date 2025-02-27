from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')


import requests
import requests as rq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from tabulate import tabulate

doge_total_cv = pd.read_csv('C:\\Users\\rjaki\\Downloads\\doge_total_contract_value.csv')
doge_saved_cv = pd.read_csv('C:\\Users\\rjaki\\Downloads\\doge_saved_contract_value.csv')
agency_budgets = pd.read_csv('C:\\Users\\rjaki\\Downloads\\agency_budgets.csv').reset_index(drop=True)
real_estate_total = pd.read_csv('C:\\Users\\rjaki\\Downloads\\real_estate_total_value.csv')
real_estate_saved = pd.read_csv('C:\\Users\\rjaki\\Downloads\\real_estate_saved.csv')

url = 'http://doge.gov/savings'
resp = rq.get(url)
html_content = resp.content

soup = BeautifulSoup(html_content, 'html.parser')
all_links = soup.find_all('a')
element_by_id = soup.find(id='element_id')
elements_by_class = soup.find_all(class_='element_class')

text_data = soup.get_text()


#Drop NA
doge_total_cv = doge_total_cv[doge_total_cv['Saved'].notna()]
doge_saved_cv = doge_saved_cv[doge_saved_cv['Saved'].notna()]
#doge_total_cv.dtypes

#String Prase Category
doge_total_cv['Description'] = doge_total_cv['Description'].astype('string')
#doge_total_cv['category'] = np.where(doge_total_cv['Description'].str.contains("DEI","DEIA","OMWI","DIVERSITY"),"DEI","Other")
doge_saved_cv['Description'] = doge_saved_cv['Description'].astype('string')
#doge_saved_cv['category'] = np.where(doge_saved_cv['Description'].str.contains("DEI","DEIA","OMWI","DIVERSITY"),"DEI","Other")
agency_budgets['agency_name'] = agency_budgets['agency_name'].str.upper()
agency_budgets['agency_name'] = agency_budgets['agency_name'].str.rsplit(n=1,expand=True)

doge_totals = pd.merge(doge_total_cv, doge_saved_cv, how='left', on=['Agency','Description','Date'])
doge_totals_agency = pd.merge(doge_totals, agency_budgets, how='left', left_on=['Agency'], right_on = ['agency_name'])

doge_totals_null_agency = doge_totals_agency[doge_totals_agency['agency_name'].isna()]
doge_totals_null_agency_nodup = doge_totals_null_agency.drop_duplicates(subset=['Agency'])
#missing_agencies = list(set(doge_totals_null_agency['agency_name'])

#Groupby
#doge_totals_agency_grp = doge_totals_agency.groupby(['Agency','agency_name','budget','pct_of_total']).sum()

doge_total_grp = doge_total_cv.groupby('Agency')['Saved'].sum()
doge_total_grp = doge_total_grp.to_frame()
#doge_cat_grp = doge_saved_cv.groupby('category')['Saved'].sum()

doge_saved_grp = doge_saved_cv.groupby('Agency')['Saved'].sum()
doge_saved_grp = doge_saved_grp.to_frame()

#Join both Grps
doge_total_saved_grp = pd.merge(doge_total_grp, doge_saved_grp, how = 'inner', on = ['Agency'])
#doge_total_saved_grp['pct_save_of_total_value'] = doge_total_saved_grp['Saved_y'] / doge_total_saved_grp['Saved_x']
doge_total_saved_grp_bg = pd.merge(doge_total_saved_grp, agency_budgets,how='left', left_on = 'Agency',right_on = 'agency_name')
doge_total_saved_grp_bg = doge_total_saved_grp_bg.rename(columns={'Saved_x':'total_contract_value','Saved_y':'amount_saved'})
doge_total_saved_grp_bg['pct_of_budget_saved'] = doge_total_saved_grp_bg['amount_saved'] / doge_total_saved_grp_bg['budget']

#clean values
doge_total_saved_grp_bg = doge_total_saved_grp_bg[doge_total_saved_grp_bg['agency_name'].notna()]

#Real Estate
doge_real_total_grp = real_estate_total.groupby('Agency')['Saved'].sum()

#Bar Charts

agency = doge_total_saved_grp_bg['agency_name'].tolist()
data = {
    'total_value': doge_total_saved_grp_bg['total_contract_value'].tolist(),
    'total_saved': doge_total_saved_grp_bg['amount_saved'].tolist(),
    'pct_of_budget': doge_total_saved_grp_bg['pct_of_budget_saved'].tolist(),
}

x = np.arange(len(agency))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()

for attribute, measurement in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('USD')
ax.set_title('DOGE Savings')
ax.set_xticks(x + width, agency)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()




