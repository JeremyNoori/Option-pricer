# Deploying to emeris.co

## Option 1: Streamlit Community Cloud (Recommended)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `appfinal.py`
5. Add secrets in the Streamlit Cloud dashboard:
   - `SUPABASE_URL` = your Supabase project URL
   - `SUPABASE_KEY` = your Supabase anon key
6. Deploy
7. In your domain registrar (for emeris.co):
   - Add a CNAME record: `@` or `www` → `your-app.streamlit.app`
   - Or use Streamlit's custom domain feature

## Option 2: Docker + VPS (Full Control)

```bash
docker build -t emeris-pricer .
docker run -p 8501:8501 \
  -e SUPABASE_URL="https://xxx.supabase.co" \
  -e SUPABASE_KEY="your-key" \
  emeris-pricer
```

Then configure nginx reverse proxy on your VPS to route emeris.co → localhost:8501.

## Option 3: Railway / Render

1. Connect GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run appfinal.py --server.port $PORT --server.headless true`
4. Add environment variables for Supabase
5. Configure custom domain emeris.co in platform settings

## Supabase Setup

1. Create project at https://supabase.com
2. Run the SQL migrations from `supabase_config.py` in the SQL Editor
3. Copy your project URL and anon key to your deployment secrets

## Domain Configuration (emeris.co)

In your DNS provider, add:
- `A` record → your server IP (if VPS)
- `CNAME` record → your-app.streamlit.app (if Streamlit Cloud)
- Enable SSL (most platforms handle this automatically)
