# GitHub & Free Deployment

## GitHub (PowerShell)
```powershell
cd job-market-intelligence
git init
git add .
git commit -m "Initial commit: job market intelligence platform"
git branch -M main
# create an EMPTY repo on github.com first, then:
git remote add origin https://github.com/<your-username>/job-market-intelligence.git
git push -u origin main
```
Do not commit private or licence-restricted datasets (check data/README.md).

## Streamlit Community Cloud (free)
1. Sign in at share.streamlit.io with GitHub.
2. **Create app** -> choose your repo, branch `main`, main file `app.py`.
3. Deploy. Dependencies install from `requirements.txt`. Share the URL.
Notes: free apps sleep when idle; upload size is limited by `.streamlit/config.toml` (50 MB). For large datasets, sample rows.
