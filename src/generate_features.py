import pandas as pd
import numpy as np
import argparse
import os

def load_google_ads(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'google_ads_campaign_stats.csv'))
    df = df.rename(columns={
        'segments_date': 'date',
        'metrics_conversions_value': 'revenue',
        'metrics_cost_micros': 'spend',
        'metrics_clicks': 'clicks',
        'metrics_impressions': 'impressions',
        'metrics_conversions': 'conversions',
        'campaign_advertising_channel_type': 'campaign_type',
        'campaign_name': 'campaign_name'
    })
    df['spend'] = df['spend'] / 1_000_000  # convert micros to dollars
    df['channel'] = 'google'
    return df[['date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'revenue', 'clicks', 'impressions', 'conversions']]

def load_bing(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'bing_campaign_stats.csv'))
    df = df.rename(columns={
        'TimePeriod': 'date',
        'Revenue': 'revenue',
        'Spend': 'spend',
        'Clicks': 'clicks',
        'Impressions': 'impressions',
        'Conversions': 'conversions',
        'CampaignType': 'campaign_type',
        'CampaignName': 'campaign_name'
    })
    df['channel'] = 'bing'
    return df[['date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'revenue', 'clicks', 'impressions', 'conversions']]

def load_meta(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'meta_ads_campaign_stats.csv'))
    df = df.rename(columns={
        'date_start': 'date',
        'conversion': 'conversions',
        'spend': 'spend',
        'clicks': 'clicks',
        'impressions': 'impressions',
        'campaign_name': 'campaign_name'
    })
    df['revenue'] = 0.0  # Meta doesn't have direct revenue column
    df['campaign_type'] = df['campaign_name'].apply(
        lambda x: 'Remarketing' if 'Remarketing' in str(x) else 'Prospecting'
    )
    df['channel'] = 'meta'
    return df[['date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'revenue', 'clicks', 'impressions', 'conversions']]

def generate_features(data_dir, out_path):
    print("Loading data...")
    google = load_google_ads(data_dir)
    bing = load_bing(data_dir)
    meta = load_meta(data_dir)

    # Combine all channels
    df = pd.concat([google, bing, meta], ignore_index=True)

    # Parse dates
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with no spend and no revenue
    df = df[(df['spend'] > 0) | (df['revenue'] > 0)]

    # Add time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['date'].dt.dayofweek

    # Add ROAS (avoid divide by zero)
    df['roas'] = df.apply(
        lambda r: r['revenue'] / r['spend'] if r['spend'] > 0 else 0, axis=1
    )

    # Aggregate to weekly level per channel + campaign_type
    weekly = df.groupby(['channel', 'campaign_type', 'year', 'week']).agg(
        spend=('spend', 'sum'),
        revenue=('revenue', 'sum'),
        clicks=('clicks', 'sum'),
        impressions=('impressions', 'sum'),
        conversions=('conversions', 'sum')
    ).reset_index()

    weekly['roas'] = weekly.apply(
        lambda r: r['revenue'] / r['spend'] if r['spend'] > 0 else 0, axis=1
    )

    print(f"Features generated: {len(weekly)} rows")
    print(f"Channels: {weekly['channel'].unique()}")
    print(f"Saving to {out_path}...")
    weekly.to_parquet(out_path, index=False)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out', default='features.parquet')
    args = parser.parse_args()
    generate_features(args.data_dir, args.out)