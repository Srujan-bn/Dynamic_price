import bcrypt
import json
import os

CREDENTIALS_FILE = "credentials.json"

# Ensure credentials file exists
if not os.path.exists(CREDENTIALS_FILE):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump({"users": {}}, file)

def load_credentials():
    """Load user credentials from the credentials file."""
    with open(CREDENTIALS_FILE, "r") as file:
        return json.load(file)

def save_credentials(credentials):
    """Save user credentials to the credentials file."""
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(credentials, file)

def hash_password(password):
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password, hashed_password):
    """Verify a password against a hashed password."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

def authenticate_user(username, password):
    """Authenticate a user based on username and password."""
    credentials = load_credentials()
    if username in credentials["users"]:
        hashed_password = credentials["users"][username]
        return verify_password(password, hashed_password)
    return False

def register_user(username, password):
    """Register a new user with username and password."""
    credentials = load_credentials()
    if username in credentials["users"]:
        return False  # Username already exists
    hashed_password = hash_password(password)
    credentials["users"][username] = hashed_password
    save_credentials(credentials)
    return True
