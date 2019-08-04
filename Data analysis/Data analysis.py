import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
customers = pd.read_csv('H:\Ecommerce prediction\Datasets\dt1.csv')

customers.head()



sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)



sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Length of Membership',kind="hex",data=customers)
sns.pairplot(customers)

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

plt.show()