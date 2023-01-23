import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('supermarket.csv')

data['Sabad']=data['Customer Id']+data['Date']



data2 = data.groupby(['Sabad','Product']).size()

data2 = pd.DataFrame(data2).reset_index()

data2.columns = ['Sabad','Product','Count']



Tdata = (data2.groupby(['Sabad', 'Product'])['Count'].sum().unstack().reset_index().fillna(0).set_index('Sabad'))

def hot_encode(x):
	if x <= 0:
		return 1
	elif x >= 1:
		return 1

Tdata_encoded = Tdata.applymap(hot_encode)

frq_items = apriori(Tdata_encoded, min_support = 0.01, use_colnames = True).sort_values('support', ascending = False)


rules = association_rules(frq_items, metric ="lift", min_threshold = 0.001)
rules = rules.sort_values(['confidence'], ascending =[False])
print(rules)