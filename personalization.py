import pandas as pd
import os
from config import LOG_FILE

def get_user_preference(username):

    if not os.path.exists(LOG_FILE):
        return None

    logs = pd.read_csv(LOG_FILE)

    if "category" not in logs.columns:
        return None

    user_logs = logs[
        (logs["user"] == username) &
        (logs["category"].notna())
    ]

    if user_logs.empty:
        return None

    return user_logs["category"].value_counts().idxmax()


def personalized_recommendations(username, df):

    preferred_category = get_user_preference(username)

    if preferred_category and "category" in df.columns:
        df = df[df["category"] == preferred_category]

    if len(df) == 0:
        return df

    return df.sample(min(5, len(df)))
