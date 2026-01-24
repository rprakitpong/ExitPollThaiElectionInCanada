import matplotlib.pyplot as plt
import numpy as np
import datetime

# Define the time range to filter out (keyboard in use)
start_time = datetime.datetime(2023, 4, 30, 11, 14, 0)
end_time = datetime.datetime(2023, 4, 30, 11, 16, 0)

# Define the word counts dictionary
word_counts = {'paethongtarn': 0, 'pita': 0, 'sudarat': 0, 'sereepisuth': 0, 'commoner': 0, 'jurin': 0, 'anutin': 0, 'prayun': 0, 'pravit': 0, 'korn': 0, 'others': 0}

# Open the file for reading
for filename in ['myfile.txt', 'myfile_sun.txt']:
    with open(filename, 'r') as f:

        # Read each line of the file
        for line in f:

            # Extract the timestamp and name from the line
            parts = line.strip().split(' ')
            timestamp = datetime.datetime.strptime(parts[0][1:] + ' ' + parts[1][:-1], '%Y-%m-%d %H:%M:%S.%f')
            name = parts[2]

            # Check if the timestamp is within the specified range
            if not (start_time <= timestamp and timestamp <= end_time):

                # Update the word count for the name in the word counts dictionary
                if name in word_counts:
                    word_counts[name] += 1

# Remove test clicks
word_counts['others'] -= 2

for word, count in word_counts.items():
    print(word, count)

# Define the colors for each word
color_map = {'paethongtarn': 'tab:red', 'pita': 'tab:orange', 'sudarat': 'tab:purple', 'sereepisuth': 'tab:olive', 'commoner': 'tab:red', 'jurin': 'tab:blue', 'anutin': 'tab:blue', 'prayun': 'tab:blue', 'pravit': 'tab:blue', 'korn': 'tab:olive', 'others': 'tab:green'}

# Define the labels for each word (names from Wikipedia)
label_map = {
    'paethongtarn': 'Pheu Thai (Paethontarn)', 
    'pita': 'Move Forward (Pita)', 
    'sudarat': 'Thai Srang Thai (Sudarat)', 
    'sereepisuth': 'Thai Liberal (Sereepisuth)', 
    'commoner': 'Commoner', 
    'jurin': 'Democrat (Jurin)', 
    'anutin': 'Bhumjaithai (Anutin)', 
    'prayun': 'UTN (Prayut)', 
    'pravit': 'Palang Pracharat (Pravit)', 
    'korn': 'Chart Pattana Kla (Korn)', 
    'others': 'Others'}

# Sort the word counts in descending order
sorted_word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the horizontal bars
bars = ax.barh(list(sorted_word_counts.keys()), list(sorted_word_counts.values()), color=[color_map[word] for word in sorted_word_counts.keys()])

# Add the value of each bar to the end of the bar
for bar in bars:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va='center')

# Set the labels for the y-axis ticks
ax.set_yticklabels([label_map[word] for word in sorted_word_counts.keys()])

ax.set_xlabel('Number of participants')
ax.set_ylabel('Party')

sum = 0
for word, count in word_counts.items():
    sum += count
plt.title('Vancouver exit poll of Thai general election 2566/2023\nTotal poll participants: ' + str(sum))

# Save the plot as a PNG file
plt.savefig('counts.png', dpi=300, bbox_inches='tight')
