import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from urllib.request import urlretrieve
import statsmodels.api as sm

es_url = 'http://www.stoxx.com/download/historical_values/hbrbcpe.txt'
vs_url = 'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'

urlretrieve(es_url, 'D:/Python for Finance/data/es.txt')
urlretrieve(vs_url, 'D:/Python for Finance/data/h_vstoxx.txt')

lines = open('D:/Python for Finance/data/es.txt', 'r').readlines()
lines = [line.replace(' ','') for line in lines]
# es.txt를 줄마다 읽어들임. 어떤 줄에는 끝에 ;이 없고 어떤 줄에는 ;이 없어서 깔끔하게 만들어야함

new_file = open('D:/Python for Finance/data/es50.txt','w')
new_file.writelines('date' + lines[3][:-1] + ';DEL' + lines[3][-1])
new_file.writelines(lines[4:])
new_file.close()

es = pd.read_csv('D:/Python for Finance/data/es50.txt', index_col = 0, parse_dates = True, sep = ';', dayfirst=True)
del es['DEL']

vs = pd.read_csv('D:/Python for Finance/data/h_vstoxx.txt', index_col = 0, header = 2, parse_dates = True, sep=',', dayfirst = True)

data = pd.DataFrame({'EUROSTOXX' : es['SX5E'][es.index > dt.datetime(1999,1,1)]})

data = data.join(pd.DataFrame({'VSTOXX' : vs['V2TX'][vs.index > dt.datetime(1999,1,1)]}))
data = data[data.index < dt.datetime(2014,9,27)]
data = data.fillna(method = 'ffill')

data.plot(subplots = True, grid = True, style = 'b', figsize=(8,6))
plt.show()
rets = np.log(data / data.shift(1))
rets.plot(subplots=True, grid=True, style='b', figsize=(8, 6))
plt.show()

rets['CONST'] = 1
rets = rets[1:]
model = sm.OLS(rets['VSTOXX'], rets[['EUROSTOXX', 'CONST']])
xdat = rets['EUROSTOXX']
ydat = rets['VSTOXX']
results = model.fit()

print(results.summary())
rets = rets[['EUROSTOXX','VSTOXX']]

plt.plot(xdat, ydat, 'r.')
ax = plt.axis()
x = np.linspace(ax[0], ax[1]+0.01)
plt.plot(x , results.params[1] + results.params[0] * x, 'b', lw=2) #lw: line width
plt.grid(True)
plt.axis('tight')
plt.xlabel('EURO STOXX 50 returns')
plt.ylabel('VSTOXX returns')
plt.show()
