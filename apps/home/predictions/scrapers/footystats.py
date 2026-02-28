"""Scraper for FootyStats football predictions."""

import logging
import re

from .base import BaseScraper
from .team_normalizer import normalize_team_name

logger = logging.getLogger(__name__)

URL = "https://footystats.org/predictions"

# Map verbose FootyStats bet descriptions to short labels
_BET_SIMPLIFICATIONS = {
    "Fulltime Result Home": "Home Win",
    "Fulltime Result Away": "Away Win",
    "Fulltime Result Draw": "Draw",
    "3Way Result 1": "Home Win",
    "3Way Result 2": "Away Win",
    "3Way Result X": "Draw",
}


def _simplify_bet(raw):
    """Shorten verbose FootyStats bet descriptions."""
    # Check exact matches first
    if raw in _BET_SIMPLIFICATIONS:
        return _BET_SIMPLIFICATIONS[raw]

    # "Goals Over/Under Over 2.5" -> "Over 2.5"
    # "Goals Over/Under Under 3.5" -> "Under 3.5"
    m = re.match(r"Goals Over/Under\s+(Over|Under)\s+(\d+\.?\d*)", raw)
    if m:
        return f"{m.group(1)} {m.group(2)} Goals"

    # "Corners Over 11.5" -> "Corners O11.5"
    m = re.match(r"Corners\s+(Over|Under)\s+(\d+\.?\d*)", raw)
    if m:
        prefix = "O" if m.group(1) == "Over" else "U"
        return f"Corners {prefix}{m.group(2)}"

    # "Both Teams to Score Yes/No" -> "BTTS Yes/No"
    if "Both Teams" in raw:
        if "Yes" in raw:
            return "BTTS Yes"
        elif "No" in raw:
            return "BTTS No"
        return "BTTS"

    return raw


class FootyStatsScraper(BaseScraper):
    SOURCE_NAME = "footystats"

    def scrape_predictions(self):
        soup = self.fetch_page(URL)
        if soup is None:
            return []

        results = []
        league_wraps = soup.find_all(class_="comp_pred_wrap")
        logger.info("FootyStats: found %d league sections", len(league_wraps))

        for wrap in league_wraps:
            league = self._parse_league_name(wrap)
            match_links = wrap.find_all("a", class_="hover-darkbg")

            for link in match_links:
                try:
                    pred = self._parse_match(link, league)
                    if pred:
                        results.append(pred)
                except Exception:
                    logger.debug("FootyStats: failed to parse a match", exc_info=True)

        logger.info("FootyStats: successfully parsed %d predictions", len(results))
        return results

    def _parse_league_name(self, wrap):
        header = wrap.find("h2")
        if not header:
            return ""
        text = header.get_text(strip=True)
        # Remove "- Predictions" suffix
        text = re.sub(r"\s*-\s*Predictions\s*$", "", text)
        return text.strip()

    def _parse_match(self, link, league):
        # --- Team names ---
        team_p = link.find("p", class_="bold")
        if not team_p:
            return None

        teams_text = team_p.get_text(strip=True)
        # Format: "Home Team vs Away Team"
        parts = re.split(r"\s+vs\s+", teams_text, maxsplit=1)
        if len(parts) != 2:
            return None

        home_raw = parts[0].strip()
        away_raw = parts[1].strip()
        if not home_raw or not away_raw:
            return None

        # --- Strength scores ---
        form_boxes = link.find_all(class_="form-box")
        home_strength = away_strength = None
        if len(form_boxes) >= 2:
            try:
                home_strength = float(form_boxes[0].get_text(strip=True))
            except (ValueError, TypeError):
                pass
            try:
                away_strength = float(form_boxes[1].get_text(strip=True))
            except (ValueError, TypeError):
                pass

        # --- Best bet prediction ---
        best_bet = None
        best_bet_odds = None
        best_p = link.find("p", class_="semi-bold")
        if best_p:
            # Extract first prediction from the best bets
            spans = best_p.find_all("span", class_="dark-gray")
            if spans:
                bet_text = spans[0].get_text(strip=True)
                # Parse "Fulltime Result Away (7.50)" or "Goals Over/Under Over 2.5 (2.03)"
                raw_bet = re.sub(r"\s*\([^)]*\)\s*$", "", bet_text).strip()
                best_bet = _simplify_bet(raw_bet)
                odds_match = re.search(r"\(([0-9.]+)\)", bet_text)
                if odds_match:
                    try:
                        best_bet_odds = float(odds_match.group(1))
                    except ValueError:
                        pass

        # Derive predicted result from best bet if it's a fulltime result prediction
        pred_result = None
        if best_bet:
            lower = best_bet.lower()
            if "home win" in lower or best_bet == "1":
                pred_result = "H"
            elif "away win" in lower or best_bet == "2":
                pred_result = "A"
            elif "draw" in lower or best_bet == "X":
                pred_result = "D"

        return {
            "source": self.SOURCE_NAME,
            "home_team": home_raw,
            "away_team": away_raw,
            "home_team_normalized": normalize_team_name(home_raw),
            "away_team_normalized": normalize_team_name(away_raw),
            "league": league,
            "date": None,  # FootyStats overview page doesn't include dates
            "time": "",
            "home_prob": None,  # Not available on overview page
            "draw_prob": None,
            "away_prob": None,
            "pred_score_home": None,
            "pred_score_away": None,
            "pred_result": pred_result,
            "best_bet": best_bet,
            "best_bet_odds": best_bet_odds,
            "_home_strength": home_strength,
            "_away_strength": away_strength,
        }
