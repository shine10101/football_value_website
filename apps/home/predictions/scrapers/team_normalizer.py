"""
Team name normalization for matching external prediction sources
to football-data.co.uk canonical team names.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Canonical name (football-data.co.uk) -> list of known aliases from external sites
TEAM_ALIASES = {
    # England - Premier League (E0)
    'Arsenal': ['Arsenal FC'],
    'Aston Villa': ['Aston Villa FC'],
    'Bournemouth': ['AFC Bournemouth', 'A.F.C. Bournemouth'],
    'Brentford': ['Brentford FC'],
    'Brighton': ['Brighton & Hove Albion', 'Brighton and Hove Albion', 'Brighton Hove Albion'],
    'Chelsea': ['Chelsea FC'],
    'Crystal Palace': ['Crystal Palace FC'],
    'Everton': ['Everton FC'],
    'Fulham': ['Fulham FC'],
    'Ipswich': ['Ipswich Town'],
    'Leicester': ['Leicester City'],
    'Liverpool': ['Liverpool FC'],
    'Man City': ['Manchester City', 'Manchester City FC', 'Man. City'],
    'Man United': ['Manchester United', 'Manchester United FC', 'Man. United', 'Man Utd'],
    'Newcastle': ['Newcastle United', 'Newcastle United FC', 'Newcastle Utd'],
    "Nott'm Forest": ['Nottingham Forest', 'Nottingham Forest FC', 'Nott Forest', "Nottm Forest"],
    'Southampton': ['Southampton FC'],
    'Tottenham': ['Tottenham Hotspur', 'Tottenham Hotspur FC', 'Spurs'],
    'West Ham': ['West Ham United', 'West Ham United FC', 'West Ham Utd'],
    'Wolves': ['Wolverhampton', 'Wolverhampton Wanderers', 'Wolverhampton Wanderers FC'],

    # England - Championship (E1)
    'Blackburn': ['Blackburn Rovers'],
    'Bristol City': ['Bristol City FC'],
    'Burnley': ['Burnley FC'],
    'Cardiff': ['Cardiff City'],
    'Coventry': ['Coventry City'],
    'Derby': ['Derby County'],
    'Hull': ['Hull City'],
    'Leeds': ['Leeds United'],
    'Luton': ['Luton Town'],
    'Middlesbrough': ['Middlesbrough FC'],
    'Millwall': ['Millwall FC'],
    'Norwich': ['Norwich City'],
    'Oxford': ['Oxford United'],
    'Plymouth': ['Plymouth Argyle'],
    'Portsmouth': ['Portsmouth FC'],
    'Preston': ['Preston North End', 'Preston NE'],
    'QPR': ['Queens Park Rangers'],
    'Sheffield Utd': ['Sheffield United'],
    'Sheffield Weds': ['Sheffield Wednesday'],
    'Stoke': ['Stoke City'],
    'Sunderland': ['Sunderland AFC'],
    'Swansea': ['Swansea City'],
    'Watford': ['Watford FC'],
    'West Brom': ['West Bromwich Albion', 'West Bromwich', 'WBA'],

    # Spain - La Liga (SP1)
    'Ath Bilbao': ['Athletic Bilbao', 'Athletic Club'],
    'Ath Madrid': ['Atletico Madrid', 'Atlético Madrid', 'Atletico de Madrid', 'Atlético de Madrid'],
    'Barcelona': ['FC Barcelona', 'Barcelona FC'],
    'Betis': ['Real Betis', 'Real Betis Balompie'],
    'Celta': ['Celta Vigo', 'Celta de Vigo', 'RC Celta'],
    'Espanol': ['Espanyol', 'RCD Espanyol'],
    'Getafe': ['Getafe CF'],
    'Girona': ['Girona FC'],
    'La Coruna': ['Deportivo La Coruña', 'Deportivo'],
    'Las Palmas': ['UD Las Palmas'],
    'Leganes': ['Leganés', 'CD Leganés'],
    'Mallorca': ['RCD Mallorca', 'Real Mallorca'],
    'Osasuna': ['CA Osasuna'],
    'Real Madrid': ['Real Madrid CF'],
    'Sevilla': ['Sevilla FC'],
    'Sociedad': ['Real Sociedad'],
    'Valencia': ['Valencia CF'],
    'Valladolid': ['Real Valladolid'],
    'Villarreal': ['Villarreal CF'],

    # Italy - Serie A (I1)
    'AC Milan': ['Milan', 'AC Milan FC'],
    'Atalanta': ['Atalanta BC'],
    'Bologna': ['Bologna FC'],
    'Cagliari': ['Cagliari Calcio'],
    'Como': ['Como 1907'],
    'Empoli': ['Empoli FC'],
    'Fiorentina': ['ACF Fiorentina'],
    'Genoa': ['Genoa CFC'],
    'Hellas Verona': ['Verona', 'Hellas Verona FC'],
    'Inter': ['Inter Milan', 'Internazionale', 'FC Internazionale'],
    'Juventus': ['Juventus FC'],
    'Lazio': ['SS Lazio', 'Lazio Roma'],
    'Lecce': ['US Lecce'],
    'Monza': ['AC Monza'],
    'Napoli': ['SSC Napoli'],
    'Parma': ['Parma Calcio', 'Parma FC'],
    'Roma': ['AS Roma'],
    'Torino': ['Torino FC'],
    'Udinese': ['Udinese Calcio'],
    'Venezia': ['Venezia FC'],

    # Germany - Bundesliga (D1)
    'Augsburg': ['FC Augsburg'],
    'B Munich': ['Bayern Munich', 'FC Bayern München', 'Bayern München', 'FC Bayern Munich', 'Bayern'],
    'Bochum': ['VfL Bochum'],
    'Dortmund': ['Borussia Dortmund', 'BVB'],
    'Ein Frankfurt': ['Eintracht Frankfurt'],
    "M'gladbach": ['Borussia Mönchengladbach', 'Monchengladbach', 'Borussia Monchengladbach', "Gladbach", "B. Monchengladbach"],
    'Freiburg': ['SC Freiburg'],
    'Heidenheim': ['1. FC Heidenheim', 'FC Heidenheim'],
    'Hoffenheim': ['TSG Hoffenheim', 'TSG 1899 Hoffenheim'],
    'Holstein Kiel': ['Kiel', 'KSV Holstein'],
    'Leverkusen': ['Bayer Leverkusen', 'Bayer 04 Leverkusen'],
    'Mainz': ['Mainz 05', 'FSV Mainz 05', '1. FSV Mainz 05'],
    'RB Leipzig': ['RasenBallsport Leipzig', 'Leipzig'],
    'St Pauli': ['FC St. Pauli', 'St. Pauli'],
    'Stuttgart': ['VfB Stuttgart'],
    'Union Berlin': ['1. FC Union Berlin', 'FC Union Berlin'],
    'Werder Bremen': ['SV Werder Bremen', 'Bremen'],
    'Wolfsburg': ['VfL Wolfsburg'],

    # France - Ligue 1 (F1)
    'Angers': ['Angers SCO'],
    'Auxerre': ['AJ Auxerre'],
    'Brest': ['Stade Brestois', 'Stade Brestois 29'],
    'Le Havre': ['Le Havre AC'],
    'Lens': ['RC Lens'],
    'Lille': ['LOSC Lille', 'Lille OSC'],
    'Lyon': ['Olympique Lyonnais', 'Olympique Lyon', 'OL'],
    'Marseille': ['Olympique Marseille', 'Olympique de Marseille', 'OM'],
    'Monaco': ['AS Monaco', 'AS Monaco FC'],
    'Montpellier': ['Montpellier HSC'],
    'Nantes': ['FC Nantes'],
    'Nice': ['OGC Nice'],
    'Paris SG': ['Paris Saint-Germain', 'PSG', 'Paris Saint Germain', 'Paris S-G'],
    'Reims': ['Stade de Reims', 'Stade Reims'],
    'Rennes': ['Stade Rennais', 'Stade Rennais FC'],
    'St Etienne': ['Saint-Étienne', 'AS Saint-Étienne', 'Saint-Etienne', 'AS St-Etienne'],
    'Strasbourg': ['RC Strasbourg', 'RC Strasbourg Alsace'],
    'Toulouse': ['Toulouse FC'],

    # Netherlands - Eredivisie (N1)
    'Ajax': ['AFC Ajax'],
    'AZ Alkmaar': ['AZ', 'AZ Alkmaar FC'],
    'Feyenoord': ['Feyenoord Rotterdam'],
    'PSV Eindhoven': ['PSV', 'PSV Eindhoven FC'],
    'Twente': ['FC Twente'],
    'Utrecht': ['FC Utrecht'],
    'Groningen': ['FC Groningen'],
    'Heerenveen': ['SC Heerenveen'],

    # Belgium - Jupiler League (B1)
    'Anderlecht': ['RSC Anderlecht'],
    'Club Brugge': ['Club Brugge KV'],
    'Genk': ['KRC Genk', 'Racing Genk'],
    'Gent': ['KAA Gent'],
    'Standard': ['Standard Liège', 'Standard Liege'],

    # Portugal - Primeira Liga (P1)
    'Benfica': ['SL Benfica'],
    'Porto': ['FC Porto'],
    'Sp Lisbon': ['Sporting CP', 'Sporting Lisbon', 'Sporting'],
    'Braga': ['SC Braga', 'Sporting Braga'],

    # Turkey - Super Lig (T1)
    'Besiktas': ['Beşiktaş', 'Besiktas JK'],
    'Fenerbahce': ['Fenerbahçe', 'Fenerbahce SK'],
    'Galatasaray': ['Galatasaray SK'],
    'Trabzonspor': ['Trabzonspor FK'],

    # Scotland - Premiership (SC0)
    'Aberdeen': ['Aberdeen FC'],
    'Celtic': ['Celtic FC'],
    'Dundee': ['Dundee FC'],
    'Hearts': ['Heart of Midlothian', 'Hearts FC'],
    'Hibernian': ['Hibernian FC', 'Hibs'],
    'Kilmarnock': ['Kilmarnock FC'],
    'Motherwell': ['Motherwell FC'],
    'Rangers': ['Rangers FC'],
    'Ross County': ['Ross County FC'],
    'St Johnstone': ['St. Johnstone', 'St Johnstone FC'],
    'St Mirren': ['St. Mirren', 'St Mirren FC'],
}

# Build reverse lookup: alias (lowercase) -> canonical name
_ALIAS_MAP = {}
for canonical, aliases in TEAM_ALIASES.items():
    _ALIAS_MAP[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = canonical

# Common suffixes to strip when attempting fuzzy normalization
_SUFFIX_PATTERN = re.compile(
    r'\s*\b(FC|CF|AFC|SC|SSC|AS|US|AC|SV|TSG|VfB|VfL|1\.|SK|FK|JK|KV|BC|CFC)\s*$',
    re.IGNORECASE,
)


def normalize_team_name(name):
    """Normalize an external team name to the football-data.co.uk canonical name."""
    if not name:
        return name

    cleaned = name.strip()

    # 1. Exact match (case-insensitive)
    if cleaned.lower() in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned.lower()]

    # 2. Strip common suffixes and retry
    stripped = _SUFFIX_PATTERN.sub('', cleaned).strip()
    if stripped and stripped.lower() in _ALIAS_MAP:
        return _ALIAS_MAP[stripped.lower()]

    # 3. No match found — log and return original
    logger.debug("Unmatched team name: '%s'", cleaned)
    return cleaned
