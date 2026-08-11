import os
import glob
import time
import argparse
import scipy.io
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Sleep stage mapping dictionary
STAGE_MAP = {
    0: 'Wake',
    1: 'Light',
    2: 'Light',
    3: 'Heavy',
    4: 'Heavy',
    5: 'Unknown'
}

def process_night(night_dir, filter_unknown=True):
    """
    Process a single night folder containing labels.mat, hr.csv, and motion.csv.
    Extracts 30-second epoch features for heart rate, accelerometry, context window, and timing.
    """
    mat_path = os.path.join(night_dir, 'labels.mat')
    hr_path = os.path.join(night_dir, 'hr.csv')
    motion_path = os.path.join(night_dir, 'motion.csv')

    if not (os.path.exists(mat_path) and os.path.exists(hr_path) and os.path.exists(motion_path)):
        return None

    try:
        # Load labels & recStart
        mat = scipy.io.loadmat(mat_path)
        rec_start_str = str(mat['recStart'][0])
        # Convert recStart from US/Eastern to UTC timestamp
        rec_start_dt = pd.to_datetime(rec_start_str).tz_localize('America/New_York')
        t_start = rec_start_dt.timestamp()

        # Sleep stage labels (expert_label preferred)
        labels = mat['expert_label'].flatten()
        n_epochs = len(labels)

        # Subject & Night identification
        parts = night_dir.replace('\\', '/').split('/')
        night_id = parts[-1]
        subject_id = parts[-2]

        # Base epochs DataFrame
        epochs_df = pd.DataFrame({
            'subject_id': subject_id,
            'night_id': night_id,
            'epoch_index': np.arange(n_epochs),
            'label_code': labels
        })

        # Load HR
        hr_df = pd.read_csv(hr_path, header=None, names=['timestamp', 'hr'])
        hr_df['epoch'] = np.floor((hr_df['timestamp'] - t_start) / 30.0).astype(int)

        # Filter HR to valid epoch bounds
        hr_df = hr_df[(hr_df['epoch'] >= 0) & (hr_df['epoch'] < n_epochs)]

        # Aggregate HR per epoch
        hr_agg = hr_df.groupby('epoch').agg(
            hr_mean=('hr', 'mean'),
            hr_std=('hr', 'std'),
            hr_min=('hr', 'min'),
            hr_max=('hr', 'max'),
            hr_count=('hr', 'count')
        ).reset_index()

        # Load Motion
        m_df = pd.read_csv(motion_path)
        m_df['vm'] = np.sqrt(m_df['x']**2 + m_df['y']**2 + m_df['z']**2)
        m_df['enorm'] = np.abs(m_df['vm'] - 1.0)
        m_df['epoch'] = np.floor((m_df['Timestamp'] - t_start) / 30.0).astype(int)

        # Filter Motion to valid epoch bounds
        m_df = m_df[(m_df['epoch'] >= 0) & (m_df['epoch'] < n_epochs)]

        # Aggregate Motion per epoch
        m_agg = m_df.groupby('epoch').agg(
            motion_vm_mean=('vm', 'mean'),
            motion_vm_std=('vm', 'std'),
            motion_vm_max=('vm', 'max'),
            motion_vm_min=('vm', 'min'),
            motion_enorm_mean=('enorm', 'mean'),
            motion_enorm_std=('enorm', 'std'),
            motion_enorm_max=('enorm', 'max'),
            motion_x_std=('x', 'std'),
            motion_y_std=('y', 'std'),
            motion_z_std=('z', 'std'),
            motion_active_count=('enorm', lambda s: (s > 0.05).sum()),
            motion_sample_count=('enorm', 'count')
        ).reset_index()

        # Merge epoch metrics
        df = pd.merge(epochs_df, hr_agg, left_on='epoch_index', right_on='epoch', how='left').drop(columns=['epoch'], errors='ignore')
        df = pd.merge(df, m_agg, left_on='epoch_index', right_on='epoch', how='left').drop(columns=['epoch'], errors='ignore')

        # Handle active ratio
        df['motion_sample_count'] = df['motion_sample_count'].fillna(0)
        df['motion_active_count'] = df['motion_active_count'].fillna(0)
        df['motion_active_ratio'] = np.where(df['motion_sample_count'] > 0, df['motion_active_count'] / df['motion_sample_count'], 0.0)

        # Forward & Backward fill for continuous signals within the night
        continuous_cols = ['hr_mean', 'hr_min', 'hr_max', 'motion_vm_mean', 'motion_vm_min', 'motion_vm_max', 'motion_enorm_mean', 'motion_enorm_max']
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Fill standard deviations and counts with 0.0 where NaN
        zero_fill_cols = ['hr_std', 'hr_count', 'motion_vm_std', 'motion_enorm_std', 'motion_x_std', 'motion_y_std', 'motion_z_std']
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # Night baseline metrics & HR normalization
        night_mean_hr = df['hr_mean'].mean()
        night_min_hr = df['hr_mean'].min()
        df['hr_diff_night_mean'] = df['hr_mean'] - night_mean_hr
        df['hr_ratio_night_mean'] = df['hr_mean'] / (night_mean_hr if night_mean_hr > 0 else 1.0)
        df['hr_diff_night_min'] = df['hr_mean'] - night_min_hr

        # Temporal / Context Window Features (Rolling windows & Lags)
        df['hr_mean_roll3'] = df['hr_mean'].rolling(window=3, center=True, min_periods=1).mean()
        df['hr_mean_roll5'] = df['hr_mean'].rolling(window=5, center=True, min_periods=1).mean()
        df['hr_std_roll5'] = df['hr_mean'].rolling(window=5, center=True, min_periods=1).std().fillna(0.0)

        df['motion_enorm_mean_roll3'] = df['motion_enorm_mean'].rolling(window=3, center=True, min_periods=1).mean()
        df['motion_enorm_mean_roll5'] = df['motion_enorm_mean'].rolling(window=5, center=True, min_periods=1).mean()
        df['motion_active_ratio_roll5'] = df['motion_active_ratio'].rolling(window=5, center=True, min_periods=1).mean()

        # Previous & Next Epoch Lag Features (t-1 and t+1 context)
        df['hr_mean_prev1'] = df['hr_mean'].shift(1).bfill()
        df['hr_mean_next1'] = df['hr_mean'].shift(-1).ffill()
        df['motion_vm_std_prev1'] = df['motion_vm_std'].shift(1).fillna(0.0)
        df['motion_vm_std_next1'] = df['motion_vm_std'].shift(-1).fillna(0.0)

        # Time progress features
        df['relative_time_in_night'] = df['epoch_index'] / float(n_epochs)
        df['time_from_start_minutes'] = (df['epoch_index'] * 30.0) / 60.0

        # Map target sleep stage label
        df['sleep_stage'] = df['label_code'].map(STAGE_MAP)

        # Filter out 'Unknown' stage (code 5) if requested
        if filter_unknown:
            df = df[df['label_code'] != 5]

        # Drop temporary label code
        df = df.drop(columns=['label_code'])

        return df

    except Exception as e:
        print(f"Error processing {night_dir}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Prepare HR/Motion Sleep Dataset for Xcode Create ML")
    parser.add_argument('--dataset_dir', type=str, default='/Users/isaque/Documents/challenges/POC/Tentativa 4/Dataset/Dataset HR:Motion',
                        help='Path to the dataset directory')
    parser.add_argument('--output_csv', type=str, default='sleep_stage_dataset.csv',
                        help='Output CSV file path for complete dataset')
    parser.add_argument('--split_test_ratio', type=float, default=0.2,
                        help='Ratio of subjects to reserve for independent test CSV (e.g. 0.2 = 20%%)')
    parser.add_argument('--filter_unknown', action='store_true', default=True,
                        help='Filter out unknown sleep stage (code 5)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    
    args = parser.parse_args()

    print(f"Scanning subject directories in: {args.dataset_dir}")
    subject_dirs = sorted(glob.glob(os.path.join(args.dataset_dir, 'Bidslab*')))
    print(f"Found {len(subject_dirs)} subject folders.")

    night_dirs = []
    for s in subject_dirs:
        for item in os.listdir(s):
            full_path = os.path.join(s, item)
            if os.path.isdir(full_path):
                night_dirs.append(full_path)

    print(f"Found {len(night_dirs)} total night recording sessions.")
    print("Extracting features per epoch across all nights...")

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_night, nd, args.filter_unknown): nd for nd in night_dirs}
        completed = 0
        for future in as_completed(futures):
            df_night = future.result()
            if df_night is not None and not df_night.empty:
                results.append(df_night)
            completed += 1
            if completed % 25 == 0 or completed == len(night_dirs):
                print(f"Processed {completed}/{len(night_dirs)} nights ({(completed/len(night_dirs))*100:.1f}%)...")

    if not results:
        print("No valid data processed!")
        return

    print("Concatenating all processed nights...")
    full_df = pd.concat(results, ignore_index=True)

    # Sort deterministically
    full_df = full_df.sort_values(by=['subject_id', 'night_id', 'epoch_index']).reset_index(drop=True)

    # Target column 'sleep_stage' first
    cols = ['sleep_stage', 'subject_id', 'night_id', 'epoch_index'] + [c for c in full_df.columns if c not in ['sleep_stage', 'subject_id', 'night_id', 'epoch_index']]
    full_df = full_df[cols]

    # Save full CSV
    output_path = os.path.abspath(args.output_csv)
    print(f"Saving full dataset ({len(full_df):,} rows, {len(cols)} columns) to: {output_path}")
    full_df.to_csv(output_path, index=False)

    # Split train/test by subject
    if args.split_test_ratio > 0:
        subjects = np.array(sorted(full_df['subject_id'].unique()))
        np.random.seed(42) # Reproducible split
        n_test = max(1, int(len(subjects) * args.split_test_ratio))
        test_subjects = set(np.random.choice(subjects, size=n_test, replace=False))
        
        train_df = full_df[~full_df['subject_id'].isin(test_subjects)].reset_index(drop=True)
        test_df = full_df[full_df['subject_id'].isin(test_subjects)].reset_index(drop=True)

        train_csv = os.path.abspath('sleep_stage_train.csv')
        test_csv = os.path.abspath('sleep_stage_test.csv')

        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print(f"Exported subject-split datasets:")
        print(f" - Train set ({len(train_df):,} rows, {len(subjects)-n_test} subjects): {train_csv}")
        print(f" - Test set  ({len(test_df):,} rows, {n_test} subjects): {test_csv}")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")
    print("\nDataset Class Distribution:")
    print(full_df['sleep_stage'].value_counts(dropna=False))

if __name__ == '__main__':
    main()
