"""Base scraper class for external prediction sources."""

import logging
from datetime import date

import cloudscraper
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for external prediction scrapers."""

    SOURCE_NAME = ""
    REQUEST_TIMEOUT = 20
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self):
        self._scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )

    def fetch_page(self, url):
        """Fetch a URL and return a BeautifulSoup object.

        Returns None if the request fails or Cloudflare blocks access.
        """
        try:
            resp = self._scraper.get(url, timeout=self.REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(
                    "%s: HTTP %s from %s", self.SOURCE_NAME, resp.status_code, url
                )
                return None
            if "Just a moment" in resp.text[:500]:
                logger.warning("%s: blocked by Cloudflare at %s", self.SOURCE_NAME, url)
                return None
            return BeautifulSoup(resp.text, "html.parser")
        except Exception:
            logger.exception("%s: failed to fetch %s", self.SOURCE_NAME, url)
            return None

    def scrape_predictions(self):
        """Scrape predictions and return a list of dicts.

        Each dict has the schema::

            {
                'source': str,
                'home_team': str,
                'away_team': str,
                'home_team_normalized': str,
                'away_team_normalized': str,
                'league': str,
                'date': date | None,
                'time': str,
                'home_prob': float | None,
                'draw_prob': float | None,
                'away_prob': float | None,
                'pred_score_home': int | None,
                'pred_score_away': int | None,
                'pred_result': str | None,
                'best_bet': str | None,
                'best_bet_odds': float | None,
            }

        Returns an empty list on failure.
        """
        raise NotImplementedError
