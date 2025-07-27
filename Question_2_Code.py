import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Files = [
    '2024-British Grand Prix-Race.csv',
    '2024-British Grand Prix-Race-weather.csv',
    '2024-Chinese Grand Prix-Race.csv',
    '2024-Chinese Grand Prix-Race-weather.csv',
    '2024-São Paulo Grand Prix-Race.csv',
    '2024-São Paulo Grand Prix-Race-weather.csv',
    '2024-Qatar Grand Prix-Race.csv',
    '2024-Qatar Grand Prix-Race-weather.csv'
]
Driver = ['ALO', 'LEC', 'PIA', 'VER']

a = pd.read_csv(Files[0])
aw = pd.read_csv(Files[1])
b = pd.read_csv(Files[2])
bw = pd.read_csv(Files[3])
d = pd.read_csv(Files[4])
dw = pd.read_csv(Files[5])
s = pd.read_csv(Files[6])
sw = pd.read_csv(Files [7])

# Fix Time Structure
# THIS CAN ONLY BE RUN ONCE AS IT DIRECTLY CHANGES THE DATA SHEET
# TO RUN THIS AGAIN RESTART THE SESSION

a['Time'] = pd.to_timedelta(a['Time'])
a['Time'] = (a['Time'].dt.total_seconds() // 60).astype(int)
a['Time'] = a['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
a = a.sort_values(by='Time', ascending=True)

aw['Time'] = pd.to_timedelta(aw['Time'])
aw['Time'] = (aw['Time'].dt.total_seconds() // 60).astype(int)
aw['Time'] = aw['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
aw = aw.sort_values(by='Time', ascending=True)

b['Time'] = pd.to_timedelta(b['Time'])
b['Time'] = (b['Time'].dt.total_seconds() // 60).astype(int)
b['Time'] = b['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
b = b.sort_values(by='Time', ascending=True)

bw['Time'] = pd.to_timedelta(bw['Time'])
bw['Time'] = (bw['Time'].dt.total_seconds() // 60).astype(int)
bw['Time'] = bw['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
bw = bw.sort_values(by='Time', ascending=True)

d['Time'] = pd.to_timedelta(d['Time'])
d['Time'] = (d['Time'].dt.total_seconds() // 60).astype(int)
d['Time'] = d['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
d = d.sort_values(by='Time', ascending=True)

dw['Time'] = pd.to_timedelta(dw['Time'])
dw['Time'] = (dw['Time'].dt.total_seconds() // 60).astype(int)
dw['Time'] = dw['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
dw = dw.sort_values(by='Time', ascending=True)

s['Time'] = pd.to_timedelta(s['Time'])
s['Time'] = (s['Time'].dt.total_seconds() // 60).astype(int)
s['Time'] = s['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
s = s.sort_values(by='Time', ascending=True)

sw['Time'] = pd.to_timedelta(sw['Time'])
sw['Time'] = (sw['Time'].dt.total_seconds() // 60).astype(int)
sw['Time'] = sw['Time'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")
sw = sw.sort_values(by='Time', ascending=True)

#Time of Day x Wind Speed
aws = aw[['Time', 'WindSpeed']]
bws = bw[['Time', 'WindSpeed']]
dws = dw[['Time', 'WindSpeed']]
sws = sw[['Time', 'WindSpeed']]

#Time of Day x Driver 1 Speed
has = a.loc[a.Driver == Driver[0], ['Time', 'SpeedI2']]
hbs = b.loc[b.Driver == Driver[0], ['Time', 'SpeedI2']]
hds = d.loc[d.Driver == Driver[0], ['Time', 'SpeedI2']]
hss = s.loc[s.Driver == Driver[0], ['Time', 'SpeedI2']]

#Time of Day x Driver 2 Speed
vas = a.loc[a.Driver == Driver[1], ['Time', 'SpeedI2']]
vbs = b.loc[b.Driver == Driver[1], ['Time', 'SpeedI2']]
vds = d.loc[d.Driver == Driver[1], ['Time', 'SpeedI2']]
vss = s.loc[s.Driver == Driver[1], ['Time', 'SpeedI2']]

#Time of Day x Driver 3 Speed
las = a.loc[a.Driver == Driver[2], ['Time', 'SpeedI2']]
lbs = b.loc[b.Driver == Driver[2], ['Time', 'SpeedI2']]
lds = d.loc[d.Driver == Driver[2], ['Time', 'SpeedI2']]
lss = s.loc[s.Driver == Driver[2], ['Time', 'SpeedI2']]

#Time of Day x Driver 4 Speed
pas = a.loc[a.Driver == Driver[3], ['Time', 'SpeedI2']]
pbs = b.loc[b.Driver == Driver[3], ['Time', 'SpeedI2']]
pds = d.loc[d.Driver == Driver[3], ['Time', 'SpeedI2']]
pss = s.loc[s.Driver == Driver[3], ['Time', 'SpeedI2']]

#Verify number of enteries for each race match

print(len(has), len(hbs), len(hds), len(hss))

print(len(vas), len(vbs), len(vds), len(vss))

print(len(las), len(lbs), len(lds), len(lss))

print(len(pas), len(pbs), len(pds), len(pss))

fig, axs = plt.subplots(2, 4, figsize=(20, 10))

sns.lineplot(x=aws['Time'], y=aws['WindSpeed'], ax=axs[0][0])
axs[0, 0].set_title(Files[0][5:-9])
axs[0, 0].set_ylabel("Wind Speed")
sns.lineplot(x=bws['Time'], y=bws['WindSpeed'], ax=axs[0][1])
axs[0, 1].set_title(Files[2][5:-9])
axs[0, 1].set_ylabel("Wind Speed")
sns.lineplot(x=dws['Time'], y=dws['WindSpeed'], ax=axs[0][2])
axs[0, 2].set_title(Files[4][5:-9])
axs[0, 2].set_ylabel("Wind Speed")
sns.lineplot(x=sws['Time'], y=sws['WindSpeed'], ax=axs[0][3])
axs[0, 3].set_title(Files[6][5:-9])
axs[0, 3].set_ylabel("Wind Speed")

sns.lineplot(x=has['Time'], y=has['SpeedI2'], label=Driver[0], ax=axs[1][0])
sns.lineplot(x=vas['Time'], y=vas['SpeedI2'], label=Driver[1], ax=axs[1][0])
sns.lineplot(x=las['Time'], y=las['SpeedI2'], label=Driver[2], ax=axs[1][0])
sns.lineplot(x=pas['Time'], y=pas['SpeedI2'], label=Driver[3], ax=axs[1][0])
axs[1, 0].set_title(Files[0][5:-9])
axs[1, 0].set_ylabel("Speed")


sns.lineplot(x=hbs['Time'], y=hbs['SpeedI2'], label=Driver[0], ax=axs[1][1])
sns.lineplot(x=vbs['Time'], y=vbs['SpeedI2'], label=Driver[1], ax=axs[1][1])
sns.lineplot(x=lbs['Time'], y=lbs['SpeedI2'], label=Driver[2], ax=axs[1][1])
sns.lineplot(x=pbs['Time'], y=pbs['SpeedI2'], label=Driver[3], ax=axs[1][1])
axs[1, 1].set_title(Files[2][5:-9])
axs[1, 1].set_ylabel("Speed")


sns.lineplot(x=hds['Time'], y=hds['SpeedI2'], label=Driver[0], ax=axs[1][2])
sns.lineplot(x=vds['Time'], y=vds['SpeedI2'], label=Driver[1], ax=axs[1][2])
sns.lineplot(x=lds['Time'], y=lds['SpeedI2'], label=Driver[2], ax=axs[1][2])
sns.lineplot(x=pds['Time'], y=pds['SpeedI2'], label=Driver[3], ax=axs[1][2])
axs[1, 2].set_title(Files[4][5:-9])
axs[1, 2].set_ylabel("Speed")


sns.lineplot(x=hss['Time'], y=hss['SpeedI2'], label=Driver[0], ax=axs[1][3])
sns.lineplot(x=vss['Time'], y=vss['SpeedI2'], label=Driver[1], ax=axs[1][3])
sns.lineplot(x=lss['Time'], y=lss['SpeedI2'], label=Driver[2], ax=axs[1][3])
sns.lineplot(x=pss['Time'], y=pss['SpeedI2'], label=Driver[3], ax=axs[1][3])
axs[1, 3].set_title(Files[6][5:-9])
axs[1, 3].set_ylabel("Speed")
