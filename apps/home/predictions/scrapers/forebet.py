"""Scraper for Forebet football predictions."""

import logging
import re
from datetime import datetime

from .base import BaseScraper
from .team_normalizer import normalize_team_name

logger = logging.getLogger(__name__)

URLS = [
    "https://www.forebet.com/en/football-tips-and-predictions-for-today",
    "https://www.forebet.com/en/football-tips-and-predictions-for-tomorrow",
]


class ForebetScraper(BaseScraper):
    SOURCE_NAME = "forebet"

    def scrape_predictions(self):
        results = []
        seen = set()

        for url in URLS:
            soup = self.fetch_page(url)
            if soup is None:
                continue

            rows = soup.find_all(class_="rcnt")
            logger.info("Forebet: found %d match rows from %s", len(rows), url)

            for row in rows:
                try:
                    pred = self._parse_row(row)
                    if pred:
                        # Deduplicate across today/tomorrow pages
                        key = (
                            pred["home_team_normalized"].lower(),
                            pred["away_team_normalized"].lower(),
                        )
                        if key not in seen:
                            seen.add(key)
                            results.append(pred)
                except Exception:
                    logger.debug("Forebet: failed to parse a row", exc_info=True)

        logger.info("Forebet: successfully parsed %d predictions", len(results))
        return results

    def _parse_row(self, row):
        # --- Team names ---
        home_span = row.find("span", class_="homeTeam")
        away_span = row.find("span", class_="awayTeam")
        if not home_span or not away_span:
            return None

        home_raw = home_span.get_text(strip=True)
        away_raw = away_span.get_text(strip=True)
        if not home_raw or not away_raw:
            return None

        # --- League code ---
        stcn = row.find(class_="stcn")
        league = stcn.get_text(strip=True) if stcn else ""

        # --- Date and time ---
        date_span = row.find("span", class_="date_bah")
        match_date = None
        match_time = ""
        if date_span:
            date_text = date_span.get_text(strip=True)
            match = re.match(r"(\d{2}/\d{2}/\d{4})\s*(\d{2}:\d{2})?", date_text)
            if match:
                try:
                    match_date = datetime.strptime(match.group(1), "%d/%m/%Y").date()
                except ValueError:
                    pass
                match_time = match.group(2) or ""

        # --- 1X2 probabilities ---
        fprc = row.find(class_="fprc")
        home_prob = draw_prob = away_prob = None
        if fprc:
            spans = fprc.find_all("span")
            probs = []
            for s in spans:
                txt = s.get_text(strip=True)
                try:
                    probs.append(int(txt))
                except (ValueError, TypeError):
                    pass
            if len(probs) >= 3:
                home_prob = probs[0]
                draw_prob = probs[1]
                away_prob = probs[2]

        # --- Predicted score ---
        predict_div = row.find(class_="predict")
        pred_score_home = pred_score_away = None
        pred_result = None
        if predict_div:
            score_span = predict_div.find("span", class_="ex_sc")
            if score_span:
                score_text = score_span.get_text(strip=True)
                score_match = re.match(r"(\d+)\s*-\s*(\d+)", score_text)
                if score_match:
                    pred_score_home = int(score_match.group(1))
                    pred_score_away = int(score_match.group(2))

        # Derive predicted result from probabilities
        if home_prob is not None and draw_prob is not None and away_prob is not None:
            max_prob = max(home_prob, draw_prob, away_prob)
            if max_prob == home_prob:
                pred_result = "H"
            elif max_prob == away_prob:
                pred_result = "A"
            else:
                pred_result = "D"

        return {
            "source": self.SOURCE_NAME,
            "home_team": home_raw,
            "away_team": away_raw,
            "home_team_normalized": normalize_team_name(home_raw),
            "away_team_normalized": normalize_team_name(away_raw),
            "league": league,
            "date": match_date,
            "time": match_time,
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "pred_score_home": pred_score_home,
            "pred_score_away": pred_score_away,
            "pred_result": pred_result,
            "best_bet": None,
            "best_bet_odds": None,
        }
