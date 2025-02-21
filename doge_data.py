from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import requests
import requests
import pandas as pd
import numpy as np
from tabulate import tabulate

doge_total_cv = pd.read_csv('C:\\Users\\rjaki\\Downloads\\doge_total_contract_value.csv')
doge_saved_cv = pd.read_csv('C:\\Users\\rjaki\\Downloads\\doge_saved_contract_value.csv')
agency_budgets = pd.read_csv('C:\\Users\\rjaki\\Downloads\\agency_budgets.csv').reset_index(drop=True)


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
doge_totals_agency_grp = doge_totals_agency.groupby(['Agency','agency_name','budget','pct_of_total']).sum()

doge_total_grp = doge_total_cv.groupby('Agency')['Saved'].sum()
doge_total_grp = doge_total_grp.to_frame()
doge_cat_grp = doge_saved_cv.groupby('category')['Saved'].sum()

doge_saved_grp = doge_saved_cv.groupby('Agency')['Saved'].sum()
doge_saved_grp = doge_saved_grp.to_frame()

#Join both Grps
doge_total_saved_grp = pd.merge(doge_total_grp, doge_saved_grp, how = 'inner', on = ['Agency'])
doge_total_saved_grp['pct_of_total_value'] = doge_total_saved_grp['Saved_y'] / doge_total_saved_grp['Saved_x']
doge_total_saved_grp_bg = pd.merge(doge_total_saved_grp,agency_budgets,how='left', left_on = 'Agency',right_on = 'agency_name')
doge_total_saved_grp_bg = doge_total_saved_grp_bg.rename(columns={'Saved_x':'total_contract_value','Saved_y':'amount_saved'})
doge_total_saved_grp_bg['pct_of_budget_saved'] = doge_total_saved_grp_bg['amount_saved'] / doge_total_saved_grp_bg['budget']

print(doge_total_cv.head())

#doge_total_cv_test = doge_total_cv[doge_total_cv]