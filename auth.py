import pandas as pd
import hashlib
import os
from config import USERS_FILE

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_admin():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame([{
            "username": "admin",
            "password": hash_password("admin123"),
            "role": "Администратор"
        }]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

def login(username, password):
    users = load_users()
    hashed = hash_password(password)
    user = users[(users["username"] == username) &
                 (users["password"] == hashed)]
    if not user.empty:
        return user.iloc[0]["role"]
    return None

def register_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False, "Пользователь уже существует"

    new_user = {
        "username": username,
        "password": hash_password(password),
        "role": "Пользователь"
    }

    users = pd.concat([users, pd.DataFrame([new_user])])
    save_users(users)
    return True, "Регистрация успешна"

def update_role(username, new_role):
    users = load_users()
    users.loc[users["username"] == username, "role"] = new_role
    save_users(users)

def delete_user(username):
    users = load_users()
    users = users[users["username"] != username]
    save_users(users)
