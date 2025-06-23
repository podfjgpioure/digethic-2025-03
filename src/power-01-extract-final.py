from pathlib import Path
import pandas as pd

print ("Importing Original household_power_consumption.txt")
data = pd.read_csv(Path("data/household_power_consumption.txt"), delimiter=";", low_memory=False)
print ("Finished Importing")

# We only want to use Global_active_power so delete all other columns
print ("Removing unused columns")
del (data['Global_reactive_power'])
del (data['Voltage'])
del (data['Global_intensity'])
del (data['Sub_metering_1'])
del (data['Sub_metering_2'])
del (data['Sub_metering_3'])

# Variable to hold all data for validated complete days
valid_data = []
# Variable to determine if we reached a new day. Initialize with Date from first row
last_date = data.at[0, 'Date']
# Temp-variable to hold the data for a single day. Until we can determine if the day is whole
last_instances = []
# Statistics
complete_days = 0
incomplete_days = 0

print ("Begin: Filtering for complete days")
for index, instance in data.iterrows():
    if instance['Date'] != last_date:
        if len(last_instances) == 1440:
            complete_days += 1
            valid_data.append(last_instances)
        else:
            incomplete_days += 1
            print (f"InComplete day: {last_date: >10}  Instances: {len(last_instances): >4} Count: {incomplete_days}")

        last_date = instance['Date']
        last_instances = []

    if len(instance['Time']) != 8:
        continue

    if instance['Global_active_power'] == '?':
        continue

    gap = float(instance['Global_active_power'])
    if gap > 0:
        last_instances.append(instance['Global_active_power'])
    else:
        print ("smaller than 0: ", gap)

print ("Finished: Filtering for complete days")
print (f"Complete days: {complete_days}   Incomplete Days: {incomplete_days}")

print ("Writing Valid-Data.")
pd.DataFrame(valid_data).to_csv(Path("data/household_power_consumption-valid.txt"), index=False)
print ("Finished")
