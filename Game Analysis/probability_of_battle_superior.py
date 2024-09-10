# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:15:36 2024

@author: dowdt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Sample data: number of trials until success (last one in-progress stage)
data = [
    {"Stage": 1, "Location": "Korea/Nagasaki", "Trials": 0},
    {"Stage": 2, "Location": "Mongolia/Saga", "Trials": 11},
    {"Stage": 3, "Location": "China/Kagoshima", "Trials": 8},
    {"Stage": 4, "Location": "Thailand/Kumamoto", "Trials": 3},
    {"Stage": 5, "Location": "Cambodia/Miyazaki", "Trials": 7},
    {"Stage": 6, "Location": "Philippines/Ōita", "Trials": 1},
    {"Stage": 7, "Location": "Japan/Fukuoka", "Trials": 3},
    {"Stage": 8, "Location": "Australia/Kochi", "Trials": 5},
    {"Stage": 9, "Location": "Singapore/Ehime", "Trials": 12},
    {"Stage": 10, "Location": "Maldives/Tokushima", "Trials": 3},
    {"Stage": 11, "Location": "India/Kagawa", "Trials": 7},
    {"Stage": 12, "Location": "Nepal/Yamaguchi", "Trials": 9},
    {"Stage": 13, "Location": "Dubai/Hiroshima", "Trials": 13},
    {"Stage": 14, "Location": "Saudi Arabia/Shimane", "Trials": 17},
    {"Stage": 15, "Location": "Kenya/Okayama", "Trials": 10},
    {"Stage": 16, "Location": "Madagascar/Tottori", "Trials": 4},
    {"Stage": 17, "Location": "South Africa/Hyōgo", "Trials": 8},
    {"Stage": 18, "Location": "Ghana/Wakayama", "Trials": 3},
    {"Stage": 19, "Location": "Sahara/Osaka", "Trials": 7},
    {"Stage": 20, "Location": "Egypt/Kyoto", "Trials": 12},
    {"Stage": 21, "Location": "Turkey/Nara", "Trials": 9},
    {"Stage": 22, "Location": "Russia/Mie", "Trials": 8},
    {"Stage": 23, "Location": "Greece/Shiga", "Trials": 4},
    {"Stage": 24, "Location": "Italy/Fukui", "Trials": 1},
    {"Stage": 25, "Location": "Monaco/Ishikawa", "Trials": 1},
    {"Stage": 26, "Location": "Spain/Aichi", "Trials": 5},
    {"Stage": 27, "Location": "France/Gifu", "Trials": 8},
    {"Stage": 28, "Location": "Germany/Toyama", "Trials": 16},
    {"Stage": 29, "Location": "Denmark/Shizuoka", "Trials": 5},
    {"Stage": 30, "Location": "Mount Aku/Mount Fuji", "Trials": 1},
    {"Stage": 31, "Location": "Norway/Yamanashi", "Trials": 1},
    {"Stage": 32, "Location": "United Kingdom/Nagano", "Trials": 3},
    {"Stage": 33, "Location": "Greenland/Niigata", "Trials": 5},
    {"Stage": 34, "Location": "Canada/Kanagawa", "Trials": 0},
    {"Stage": 35, "Location": "New York/Chiba", "Trials": 0},
    {"Stage": 36, "Location": "Bermuda/Tokyo", "Trials": 0},
    {"Stage": 37, "Location": "Jamaica/Saitama", "Trials": 0},
    {"Stage": 38, "Location": "Colombia/Gunma", "Trials": 0},
    {"Stage": 39, "Location": "Brazil/Tochigi", "Trials": 0},
    {"Stage": 40, "Location": "Argentina/Ibaraki", "Trials": 0},
    {"Stage": 41, "Location": "Machu Picchu/Fukushima", "Trials": 0},
    {"Stage": 42, "Location": "Easter Island/Miyagi", "Trials": 0},
    {"Stage": 43, "Location": "Mexico/Yamagata", "Trials": 0},
    {"Stage": 44, "Location": "NASA/Iwate", "Trials": 0},
    {"Stage": 45, "Location": "Las Vegas/Akita", "Trials": 0},
    {"Stage": 46, "Location": "Hollywood/Aomori", "Trials": 0},
    {"Stage": 47, "Location": "Alaska/Hokkaido", "Trials": 0},
    {"Stage": 48, "Location": "Hawaii/Okinawa", "Trials": 0},
    {"Stage": 49, "Location": "Moon/Iriomote", "Trials": 0},
    {"Stage": "IN", "Location": "Mount Aku/Mount Fuji Invasion", "Trials": 0},
]

df = pd.DataFrame(data)
# Find the index of the next row with 0 trials after the 2nd row
next_zero_index = df.index[(df.index > 1) & (df['Trials'] == 0)].min()

# Extract trials between 2nd row and the next row with 0 trials
df_subset = df.loc[2:next_zero_index - 1]
trials_series = df_subset['Trials']


# Convert to pandas Series for easier analysis
# trials_series = df[df['Trials'] != 0]['Trials']

# Calculate mean and standard deviation
mean = trials_series.mean()
std = trials_series.std()

# Generate x values for plotting
x = np.linspace(0, max(trials_series) * 2, 1000)

# Calculate normal distribution
y = stats.norm.pdf(x, mean, std)

# Plotting
plt.figure(figsize=(10, 8))

# Bell curve plot
plt.subplot(3, 2, 1)
plt.plot(x, y, 'b-')
plt.xlabel('Number of Trials')
plt.ylabel('Probability Density')
plt.title('Probability of Success at Each Trial Number')
plt.grid(True)
plt.axvline(mean, color='r', linestyle='--',
            label=f'Expected Value ({mean:.2f})')
plt.axvline(mean - std, color='g', linestyle=':',
            label=f'±1 Std Dev ({std:.2f})')
plt.axvline(mean + std, color='g', linestyle=':')
plt.legend()

# Cumulative probability plot (CDF)
plt.subplot(3, 2, 2)
cdf = stats.norm.cdf(x, mean, std)
plt.plot(x, cdf, 'g-')
plt.xlabel('Number of Trials')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Probability of Success')
plt.grid(True)

# Q-Q plot
plt.subplot(3, 2, 3)
stats.probplot(trials_series, dist="norm", plot=plt)
plt.title('Q-Q Plot')

# Box plot
plt.subplot(3, 2, 4)
plt.boxplot(trials_series)
plt.title('Box Plot')
plt.ylabel('Number of Trials')

# Calculate the probability of success from the mean of trials
p = 1 / np.mean(trials_series)


# Histogram
plt.subplot(3, 2, 5)
# Calculate the range for the bins
max_attempts = trials_series.max()
min_attempts = trials_series.min()

# Create bins from min to max+1 with step 1 (to ensure each integer has its own bin)
bins = np.arange(min_attempts, max_attempts + 2, 1) - 0.5

# Plot the histogram
counts, bins, _ = plt.hist(trials_series, bins=bins, align='mid',
                           rwidth=0.8, density=True, alpha=0.7, label='Observed')

plt.title('Distribution of Attempts Until Success')
plt.xlabel('Number of Attempts')
plt.ylabel('Probability')

# Set x-ticks to be integers
plt.xticks(range(int(min_attempts), int(max_attempts) + 1))

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Overlay geometric distribution
x = np.arange(min_attempts, max_attempts + 1)
geometric_dist = stats.geom(p)
plt.plot(x, geometric_dist.pmf(x), 'r-', lw=2, label='Geometric Distribution')

plt.legend()


plt.tight_layout()
plt.show()

# Calculate quartiles
lower_quartile = np.percentile(trials_series, 25)
median = np.median(trials_series)
upper_quartile = np.percentile(trials_series, 75)

# Print statistics and probabilities
print(f"Mean (Expected Value): {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
print(f"Lower Quartile: {lower_quartile:.2f}")
print(f"Median: {median:.2f}")
print(f"Upper Quartile: {upper_quartile:.2f}")
print(f"Maximum attempts: {max_attempts}")

print("-" * 50)

# Calculate the empirical drop chance
total_attempts = trials_series.sum()
total_successes = len(trials_series)
empirical_drop_chance = total_successes / total_attempts
print(f"Empirical drop chance: {empirical_drop_chance:.4f}")
print(f"Empirical drop chance percentage: {empirical_drop_chance * 100:.2f}%")
print("-" * 50)


print("Probability Estimates:")
print(
    f"Probability of success by lower quartile ({lower_quartile:.2f} trials): {stats.norm.cdf(lower_quartile, mean, std):.4f}")
print(
    f"Probability of success by median ({median:.2f} trials): {stats.norm.cdf(median, mean, std):.4f}")
print(
    f"Probability of success by upper quartile ({upper_quartile:.2f} trials): {stats.norm.cdf(upper_quartile, mean, std):.4f}")
print("-" * 50)
