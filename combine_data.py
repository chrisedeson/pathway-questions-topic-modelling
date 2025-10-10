#!/usr/bin/env python3
import pandas as pd
import random
from datetime import datetime, timedelta
import uuid

def generate_timestamps(start_date, end_date, num_timestamps):
    """Generate random timestamps between start_date and end_date in descending order"""
    timestamps = []
    start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

    for _ in range(num_timestamps):
        # Generate random time between start and end
        time_diff = (end - start).total_seconds()
        random_seconds = random.uniform(0, time_diff)
        random_time = start + timedelta(seconds=random_seconds)
        timestamps.append(random_time)

    # Sort in descending order (newest first)
    timestamps.sort(reverse=True)

    # Format as the same format as langfuse: "2025-10-08 08:28:09.053000+00:00"
    return [t.strftime('%Y-%m-%d %H:%M:%S.%f') + '+00:00' for t in timestamps]

def main():
    # Read the incomplete.csv file
    print("Reading incomplete.csv...")
    with open('/home/chrisflex/byu-pathway/pathway-questions-topic-modelling/notebook/incomplete.csv', 'r', encoding='utf-8') as f:
        incomplete_lines = f.readlines()

    print(f"Found {len(incomplete_lines)} lines in incomplete.csv")

    # Read the existing langfuse file
    print("Reading langfuse_traces_10_08_25.csv...")
    langfuse_df = pd.read_csv('/home/chrisflex/byu-pathway/pathway-questions-topic-modelling/notebook/langfuse_traces_10_08_25.csv')
    print(f"Found {len(langfuse_df)} rows in langfuse file")

    # Get the latest timestamp from langfuse file
    latest_timestamp = langfuse_df['timestamp'].max()
    print(f"Latest timestamp in langfuse file: {latest_timestamp}")

    # Generate timestamps for incomplete data (July 1st to latest date)
    start_date = "2025-07-01T00:00:00.000Z"
    end_date = latest_timestamp
    timestamps = generate_timestamps(start_date, end_date, len(incomplete_lines))
    print(f"Generated {len(timestamps)} timestamps")

    # Create new rows for incomplete data
    new_rows = []
    for i, line in enumerate(incomplete_lines):
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Generate a unique ID
        unique_id = str(uuid.uuid4())

        # Create row with 24 columns (matching langfuse format)
        # Columns: bookmarked,createdAt,environment,externalId,html_path,id,input,latency,metadata,name,observations,output,projectId,public,release,scores,session_id,tags,timestamp,total_cost,updatedAt,user_feedback,user_id,version
        row = [
            'False',  # bookmarked
            '',       # createdAt (empty)
            'default', # environment
            '',       # externalId (empty)
            '',       # html_path (empty)
            unique_id, # id
            line.strip('"'),  # input (remove surrounding quotes if any)
            '0.0',    # latency
            '{}',     # metadata (empty dict)
            'chat',   # name
            '[]',     # observations (empty array)
            '',       # output (empty)
            'cm33dfx4404qhurr6i9n8hldc',  # projectId
            'False',  # public
            'af547eb1a2d5b0ee9b7bd5f8d23b15b15ecfdc25',  # release
            '[]',     # scores (empty array)
            '',       # session_id (empty)
            '[]',     # tags (empty array)
            timestamps[i],  # timestamp
            '0.0',    # total_cost
            '',       # updatedAt (empty)
            '',       # user_feedback (empty)
            '',       # user_id (empty)
            ''        # version (empty)
        ]
        new_rows.append(row)

    print(f"Created {len(new_rows)} new rows")

    # Convert to DataFrame
    new_df = pd.DataFrame(new_rows, columns=langfuse_df.columns)

    # Combine the DataFrames
    combined_df = pd.concat([langfuse_df, new_df], ignore_index=True)

    # Sort by timestamp in descending order (newest first)
    # Convert to datetime for sorting, then back to string
    combined_df['timestamp_dt'] = pd.to_datetime(combined_df['timestamp'], utc=True, format='mixed')
    combined_df = combined_df.sort_values('timestamp_dt', ascending=False)
    combined_df['timestamp'] = combined_df['timestamp_dt'].dt.strftime('%Y-%m-%d %H:%M:%S.%f') + '+00:00'
    combined_df = combined_df.drop('timestamp_dt', axis=1)

    print(f"Combined dataset has {len(combined_df)} rows")

    # Write to new file
    output_file = '/home/chrisflex/byu-pathway/pathway-questions-topic-modelling/notebook/langfuse_traces_combined.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Written combined data to {output_file}")

if __name__ == "__main__":
    main()
