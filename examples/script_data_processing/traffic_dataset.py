import numpy as np
import pandas as pd
import datetime
import pickle


def datetime_to_hours(dt: datetime.timedelta):
    # convert datetime to hours
    return round(dt.total_seconds() / 3600, ndigits=8)


def format_data_test(seq_len: datetime.timedelta, df: pd.DataFrame, k_events: int):
    cur_seq_idx = 0
    total_seqs = []

    # iterate through each sequence id which contains multiple events
    for seq_id in df["seq_id"].unique():
        cur_seq = []
        last_event_time = datetime.timedelta()
        last_same_event = [datetime.timedelta() for _ in range(k_events)]
        # iterate through each event in current sequence
        j = 1
        offset = datetime.timedelta()
        delta = datetime.timedelta(seconds=1)
        for i, row in df[df["seq_id"] == seq_id].iterrows():
            diff = (
                row["timediff_start"] % seq_len
            ) + offset # get time difference from start of week
            if datetime_to_hours(diff - last_event_time) == 0.0:
                offset += delta
                diff += delta
                
            cur_seq.append(
                {  # append event dictionary to sequence
                    "idx_event": j,
                    "time_since_last_event": datetime_to_hours(diff - last_event_time),
                    "time_since_last_same_event": datetime_to_hours(
                        diff - last_same_event[row["severity_level"]]
                    ),
                    "type_event": row["severity_level"],
                    "time_since_start": datetime_to_hours(diff),
                }
            )
            last_same_event[row["severity_level"]] = diff
            last_event_time = diff
            j += 1

        total_seqs.append(cur_seq)

    return total_seqs


def data_split(
    df: pd.DataFrame,
    train: float,
    test: float,
    dev: float,
    seq_len: datetime.timedelta,
    k_events,
    seed=None,
):
    """
    split dataframe based on train, test, dev percentages
    keeps events that are in the same sequence together when splitting into train, test, dev
    """
    if seed:
        np.random.seed(seed)

    if seq_len is None:
        seq_len = datetime.timedelta(days=5000)
    # assign each sequence a unique id by dividing the time since start by the sequence length and rounding down
    df["timediff_start"] = pd.to_timedelta(df["timediff_start"])
    seq_id = df["timediff_start"] // seq_len
    df["seq_id"] = df["timediff_start"] // seq_len
    periods = seq_id.unique()
    np.random.shuffle(periods)

    print(df.value_counts("seq_id"))

    # calculate splits and assign the calculated number of periods to corresponding dataset
    n = len(periods)
    train_split = int(train * n)
    test_split = int(test * n)

    train_period = periods[:train_split]
    test_period = periods[train_split : train_split + test_split]
    dev_period = periods[train_split + test_split :]

    train_df = df[(df["timediff_start"] // seq_len).isin(train_period)]
    test_df = df[(df["timediff_start"] // seq_len).isin(test_period)]
    dev_df = df[(df["timediff_start"] // seq_len).isin(dev_period)]

    return (
        format_data_test(seq_len, train_df, k_events),
        format_data_test(seq_len, test_df, k_events),
        format_data_test(seq_len, dev_df, k_events),
    )


def write_data(k_events, csv_path, data_out_path, train, test, dev, seq_len):
    df = pd.read_csv(csv_path)
    train_dict, test_dict, dev_dict = data_split(df, train, test, dev, seq_len, k_events, seed=1234) # use weeks as sequence
    print("number of sequences:", len(train_dict), len(test_dict), len(dev_dict))

    # use pickle to export data
    with open(f"{data_out_path}/train.pkl", "wb") as out:
        data_dict = {"dim_process": k_events, "train": train_dict}
        pickle.dump(data_dict, out, pickle.HIGHEST_PROTOCOL)

    with open(f"{data_out_path}/test.pkl", "wb") as out:
        data_dict = {"dim_process": k_events, "test": test_dict}
        pickle.dump(data_dict, out, pickle.HIGHEST_PROTOCOL)

    with open(f"{data_out_path}/dev.pkl", "wb") as out:
        data_dict = {"dim_process": k_events, "dev": dev_dict}
        pickle.dump(data_dict, out, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    k_events = 3
    csv_path = "../data/traffic_raw/CleanCrashData.csv"
    data_out_path = "../data/"
    write_data(k_events, csv_path, data_out_path + "traffic_week", 0.8, 0.1, 0.1, datetime.timedelta(days=7))
    write_data(k_events, csv_path, data_out_path + "traffic_all", 1.0, 0, 0, None)
