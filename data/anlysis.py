# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('checkins-4sq.txt', sep='\t',
                 names=['user_id', 'timestamp', 'latitude', 'longitude', 'poi_id'])

print("=" * 60)
print("Dataset Statistics Analysis")
print("=" * 60)

# Original dataset statistics
print("\n[BEFORE FILTERING]")
print(f"Original #Users: {df['user_id'].nunique():,}")
print(f"Original #Check-ins: {len(df):,}")

# Filter: only keep users with >= 101 check-ins
user_checkin_counts = df['user_id'].value_counts()
valid_users = user_checkin_counts[user_checkin_counts >= 101].index
df = df[df['user_id'].isin(valid_users)]

print(f"\n[AFTER FILTERING] (users with >= 101 check-ins)")
print(f"Filtered out {len(user_checkin_counts) - len(valid_users):,} users")
print(f"Remaining users: {len(valid_users):,}")
print(f"Remaining check-ins: {len(df):,}")

# Basic statistics
n_users = df['user_id'].nunique()
n_pois = df['poi_id'].nunique()
n_checkins = len(df)

print("\n" + "=" * 60)
print("1. Basic Statistics:")
print("=" * 60)
print(f"   #Users: {n_users:,}")
print(f"   #POIs: {n_pois:,}")
print(f"   #Check-ins: {n_checkins:,}")

# Calculate density
density = n_checkins / (n_users * n_pois)
print(f"\n2. Density:")
print(f"   Density: {density:.6f}")

# POI frequency statistics
poi_freq = df['poi_id'].value_counts()

# POI frequency < 200
pois_below_200 = (poi_freq < 200).sum()
percentage_below_200 = (pois_below_200 / n_pois) * 100

# POI frequency < 100
pois_below_100 = (poi_freq < 100).sum()
percentage_below_100 = (pois_below_100 / n_pois) * 100

print(f"\n3. POI Frequency Distribution:")
print(f"   POI frequency <200 (%): {percentage_below_200:.2f}%")
print(f"   POI frequency <100 (%): {percentage_below_100:.2f}%")
print(f"   POI frequency <50 (%): {((poi_freq < 50).sum() / n_pois * 100):.2f}%")

# Calculate trajectories (segment_length=20 based on paper)
trajectory_length = 20
n_trajectories = 0

# Group by user and calculate trajectories
for user_id, group in df.groupby('user_id'):
    user_checkins = len(group)
    n_trajectories += user_checkins // trajectory_length

print(f"\n4. Trajectory Statistics (segment_length={trajectory_length}):")
print(f"   #Trajectories: {n_trajectories:,}")

# Time range
df['timestamp'] = pd.to_datetime(df['timestamp'])
duration = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
print(f"\n5. Time Range:")
print(f"   Duration: {duration}")

# Additional statistics
print(f"\n6. Additional Statistics:")
print(f"   Avg check-ins per user: {n_checkins / n_users:.2f}")
print(f"   Avg visits per POI: {n_checkins / n_pois:.2f}")
print(f"   Max POI visits: {poi_freq.max():,}")
print(f"   Min POI visits: {poi_freq.min()}")
print(f"   Min user check-ins: {user_checkin_counts[valid_users].min()}")
print(f"   Max user check-ins: {user_checkin_counts[valid_users].max()}")

# Top 20 most popular POIs
print(f"\n7. Top 20 Most Popular POIs:")
print(poi_freq.head(20))

# Table 1 format output
print("\n" + "=" * 60)
print("Table 1 Format Output:")
print("=" * 60)
print(f"Dataset                    Wuhan")
print(f"Duration                   {duration}")
print(f"#Users                     {n_users:,}")
print(f"#POIs                      {n_pois:,}")
print(f"#Check-ins                 {n_checkins:,}")
print(f"#Trajectories              {n_trajectories:,}")
print(f"Density                    {density:.6f}")
print(f"POI frequency <200 (%)     {percentage_below_200:.2f}%")
print(f"POI frequency <100 (%)     {percentage_below_100:.2f}%")
print("=" * 60)
