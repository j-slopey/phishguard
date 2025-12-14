# PhishGuard

A machine learning-based phishing email detection system using XGBoost.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/j-slopey/phishguard.git
cd phishguard
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook to build model.

## Usage

1. Store emails in a JSON file with the following format (A sample email from each class is included):
```json
{
    "sender": "...",
    "receiver": "...",
    "date": "...",
    "subject": "...",
    "body": "..."
}
```

2. Run the included classification script:
```bash
python phishguard_predict.py sample_phish.json
```

## Credits

### Dataset

This project uses this Phishing Email Dataset available on Kaggle:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

*Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619*

### Email Domains List

This project uses a list of free email domains from the [free-email-domains](https://github.com/Kikobeats/free-email-domains/tree/409a772efec87ed4c4d9b2e3b67bae881869ba7f) repository by Kiko Beats.


