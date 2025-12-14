import numpy as np
import json
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper classifier to apply a custom threshold to predicted probabilities."""
    def __init__(self, estimator, threshold=None):
        self.custom_threshold = True if threshold is not None else False
        self.estimator_ = estimator
        self.threshold = threshold if threshold is not None else 0.5
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        result = (probs >= self.threshold).astype(int)
        if not self.custom_threshold:
            print("No custom threshold has been set for this model. Using 0.5")
        return result
    
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input data to get propper feature set."""

    data['receiver'] = data['receiver'].str.replace('undisclosed-recipients:;', 'Unknown')
    # -- Timestamp features -- 
    data['date_parsed'] = pd.to_datetime(data['date'], errors='coerce', utc=True)

    # Week-of-year
    iso_week = data['date_parsed'].dt.isocalendar().week
    iso_week = iso_week.astype(float)
    week0 = ((iso_week - 1) % 52)
    theta_week = 2.0 * np.pi * week0 / 52
    data['sin_week'] = np.where(week0.notna(), np.sin(theta_week), 0.0)
    data['cos_week'] = np.where(week0.notna(), np.cos(theta_week), 0.0)

    # Hour-of-day 
    hour = data['date_parsed'].dt.hour.astype(float)  # NaN for missing
    theta_hour = 2.0 * np.pi * hour / 24
    data['sin_hour'] = np.where(hour.notna(), np.sin(theta_hour), 0.0)
    data['cos_hour'] = np.where(hour.notna(), np.cos(theta_hour), 0.0)

    # Weekend binary (0/1)
    weekday = data['date_parsed'].dt.weekday
    data['is_weekend'] = np.where(weekday.isna(), 0, ((weekday >= 5).astype(int)))


    # -- Sender/reciever feature engineering -- 
    with open('free_domains.json', 'r') as file:
        public_email_domains = json.load(file)
        
    email_regex = r'([a-zA-Z0-9._%+\-|{}^&"\'=]+@(?:[a-zA-Z0-9.-]+|\[[0-9.]+\]))'    
    for column_name in ('sender', 'receiver'):
        data[f'{column_name}_email'] = data[column_name].str.extract(email_regex, expand=False)
        data[f'{column_name}_domain'] = data[f'{column_name}_email'].str.split('@', n=1).str[1]
        data[f'{column_name}_domain_len'] = data[f'{column_name}_domain'].str.len()
        data[f'{column_name}_domain_public'] = data[f'{column_name}_domain'].str.lower().isin(public_email_domains).astype(int)
        data[f'{column_name}_n_subdomains'] = data[f'{column_name}_domain'].str.lower().str.count(r'\.')
        data[f'{column_name}_email_n_digits'] = data[f'{column_name}_domain'].str.lower().str.count(r'\d')
        
        data[f'{column_name}_name'] = data[column_name].str.replace(email_regex, '', regex=True)
        data[f'{column_name}_name'] = data[f'{column_name}_name'].str.replace(r'[<>"\'\(\)]', '', regex=True).str.strip()
        

    data['sender_name_contains_email'] = data['sender_name'].str.contains('@', na=False).astype(int)
    data["subject_body"] = data["subject"].fillna("") + " " + data["body"].fillna("")
    data[['body', 'subject']] = data[['body', 'subject']].fillna('Unknown')

    # Timestamp feature list
    final_columns = [
        'subject',
        'body',
        'subject_body',
        "sin_week",
        "cos_week",
        "sin_hour",
        "cos_hour",
        "is_weekend",
        "sender_domain_public",
        "sender_domain_len",
        "sender_n_subdomains",
        "sender_email_n_digits",
        "sender_name_contains_email",
    ]
    
    return data[final_columns]

def main():
    pass

if __name__ == "__main__":
    main()  