import pandas as pd
import numpy as np
import pickle
import argparse
import os
from sklearn.linear_model import LinearRegression

def train_models(features_path, model_path):
    print("Loading features...")
    df = pd.read_parquet(features_path)

    models = {}
    channels = df['channel'].unique()

    for channel in channels:
        ch_df = df[df['channel'] == channel].copy()
        ch_df = ch_df[ch_df['spend'] > 0]

        if len(ch_df) < 5:
            print(f"Skipping {channel} — not enough data")
            continue

        X = ch_df[['spend', 'month']].values if 'month' in ch_df.columns else ch_df[['spend']].values
        y_revenue = ch_df['revenue'].values
        y_roas = ch_df['roas'].values

        rev_model = LinearRegression()
        rev_model.fit(X, y_revenue)

        roas_model = LinearRegression()
        roas_model.fit(X, y_roas)

        # Store residuals for uncertainty estimation
        rev_residuals = y_revenue - rev_model.predict(X)
        roas_residuals = y_roas - roas_model.predict(X)

        models[channel] = {
            'revenue_model': rev_model,
            'roas_model': roas_model,
            'rev_residuals': rev_residuals,
            'roas_residuals': roas_residuals,
            'avg_weekly_spend': ch_df['spend'].mean()
        }
        print(f"Trained model for: {channel}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {model_path}")
    return models

def forecast(models, horizon_days=30, budget_multiplier=1.0, n_simulations=1000):
    weeks = max(1, horizon_days // 7)
    results = []

    for channel, m in models.items():
        weekly_spend = m['avg_weekly_spend'] * budget_multiplier
        month = 6  # current month as default

        n_features = m['revenue_model'].n_features_in_
        if n_features == 2:
            X_pred = np.array([[weekly_spend, month]])
        else:
            X_pred = np.array([[weekly_spend]])

        rev_pred = m['revenue_model'].predict(X_pred)[0]
        roas_pred = m['roas_model'].predict(X_pred)[0]

        # Bootstrap for uncertainty
        rev_samples = [
            (rev_pred + np.random.choice(m['rev_residuals'])) * weeks
            for _ in range(n_simulations)
        ]
        roas_samples = [
            max(0, roas_pred + np.random.choice(m['roas_residuals']))
            for _ in range(n_simulations)
        ]

        results.append({
            'channel': channel,
            'horizon_days': horizon_days,
            'predicted_spend': round(weekly_spend * weeks, 2),
            'revenue_low': round(np.percentile(rev_samples, 10), 2),
            'revenue_mid': round(np.percentile(rev_samples, 50), 2),
            'revenue_high': round(np.percentile(rev_samples, 90), 2),
            'roas_low': round(np.percentile(roas_samples, 10), 2),
            'roas_mid': round(np.percentile(roas_samples, 50), 2),
            'roas_high': round(np.percentile(roas_samples, 90), 2),
        })

    return pd.DataFrame(results)

def run_predict(features_path, model_path, output_path):
    # Train if model doesn't exist
    if not os.path.exists(model_path):
        print("No model found, training now...")
        models = train_models(features_path, model_path)
    else:
        print("Loading existing model...")
        with open(model_path, 'rb') as f:
            models = pickle.load(f)

    all_results = []
    for horizon in [30, 60, 90]:
        df = forecast(models, horizon_days=horizon)
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(final.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='features.parquet')
    parser.add_argument('--model', default='./pickle/model.pkl')
    parser.add_argument('--output', default='./output/predictions.csv')
    args = parser.parse_args()
    run_predict(args.features, args.model, args.output)