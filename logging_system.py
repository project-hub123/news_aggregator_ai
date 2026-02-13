import pandas as pd
import os
from datetime import datetime
from config import LOG_FILE

def log_action(username, action_type, input_text, result, category=None):

    log_data = {
        "timestamp": datetime.now(),
        "user": username,
        "action": action_type,
        "input": str(input_text)[:300],
        "result": str(result)[:300],
        "category": category
    }

    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        logs = pd.concat([logs, pd.DataFrame([log_data])])
    else:
        logs = pd.DataFrame([log_data])

    logs.to_csv(LOG_FILE, index=False)

def get_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame()
