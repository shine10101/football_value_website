# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply migrations
python manage.py migrate

# Run dev server (http://127.0.0.1:8000/)
python manage.py runserver

# Collect static files
python manage.py collectstatic --noinput

# Run tests
python manage.py test
python manage.py test apps.home        # single app
```

### Docker / Production

```bash
# Build and run with Docker Compose (uses Traefik for TLS)
docker compose up -d --build

# View logs
docker compose logs -f app
```

## Architecture

This is a **Django 4.2** web app built on the [Black Dashboard](https://www.creative-tim.com/product/black-dashboard-django) template. Its core purpose is displaying football (soccer) betting value predictions.

### Configuration (`core/`)
- `settings.py` — reads all config from `.env` via `django-environ`. Key env vars: `SECRET_KEY`, `DEBUG`, `DOMAIN`, `DATA_DIR`, `PREDICTIONS_CSV`.
- `DATA_DIR` sets the base path for both `db.sqlite3` and `predictions.csv`; the Docker volume `app-data` is mounted at `/app/data`.
- `PREDICTIONS_CSV` can be an absolute path or relative to `DATA_DIR`.

### Apps

**`apps/home/`** — the main application, all pages require login.
- `models.py` — `Prediction` model stores historical predictions with actual results for performance tracking. Fields include model predictions (`h_win`, `draw`, `a_win`, `pred_ftr`), bookmaker odds (`odds_h/d/a`), actual results (`actual_ftr/fthg/ftag`), and a `resolved` flag. Unique on `(div, date, home_team, away_team)`.
- `views.py` — loads `predictions.csv` into a module-level mtime-based cache (`_csv_cache`). The prediction pipeline can be triggered via `POST /api/refresh/` which spawns a background thread; `GET /api/refresh/status/` polls progress. Includes a `performance` view with detailed accuracy, calibration, financial P/L simulation (flat-stake), threshold analysis, and Kelly criterion calculations.
- `predictions/` — the ML pipeline:
  - `data_ingestion.py` — fetches historical match data and upcoming fixtures from external sources.
  - `poisson_analysis.py` — Poisson model to generate match outcome probabilities (HWin, Draw, AWin).
  - `calculate_value.py` — computes betting value (`Max_Value`, `Max_Value_Result`) by comparing model probabilities to implied odds.
  - `pipeline.py` — orchestrates the full pipeline, saves output to `PREDICTIONS_CSV`. Also archives predictions to the database via `archive_predictions()` and resolves past predictions against actual results via `resolve_results()`.
- `management/commands/` — custom Django management commands:
  - `resolve_results.py` — `python manage.py resolve_results` fetches historical data and resolves unresolved predictions.
  - `backcast.py` — `python manage.py backcast [--leagues E0 E1] [--season 2526] [--min-games 10]` walks historical matches chronologically, generating predictions using only data available at the time, and saves them as resolved `Prediction` objects for the performance page.

**`apps/authentication/`** — session-based login/register (Django built-in auth).

### Templates (`apps/templates/`)
- `layouts/base.html` — base layout for authenticated pages (Black Dashboard theme).
- `includes/sidebar.html` — navigation sidebar with links to Dashboard, Value Table, Performance, and Logout.
- `home/index.html` — dashboard with Chart.js charts (doughnut for prediction breakdown, bar for league counts and average value).
- `home/tables.html` — paginated predictions table with league filter dropdown.
- `home/performance.html` — performance analysis page with accuracy stats, calibration charts, financial P/L simulation, threshold comparison, and team/league breakdowns.

### Static files
- Source files in `apps/static/assets/`; collected to `core/staticfiles/` for production (served by WhiteNoise).

### URL routing
`core/urls.py` → authentication routes first, then `apps/home/urls.py` last (which has a catch-all `re_path` for template-based pages).

### Deployment
- Docker + Gunicorn (port 5005) behind **Traefik** for TLS termination. See `docker-compose.yml` and `deployment/traefik/`.
- `.env.production` holds production secrets; `.env` is used for local dev.
- `entrypoint.sh` runs `migrate`, `collectstatic`, then starts Gunicorn.

### Key data flow
1. `predictions.csv` is pre-generated (or refreshed via `/api/refresh/`) and read directly by views — no database tables for prediction data.
2. The CSV columns used: `Div`, `Date`, `Time`, `HomeTeam`, `AwayTeam`, `HWin`, `Draw`, `AWin`, `Pred_FTR`, `Max_Value`, `Max_Value_Result`, `Pred_FTHG`, `Pred_FTAG`.
3. `Max_Value > 0` indicates positive expected value bets; data is pre-sorted by `Max_Value` descending before display.
