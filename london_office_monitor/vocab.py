"""The fixed vocabulary shared by the seed files, the database constraints, the MCP tool schemas and the agent."""
from enum import StrEnum


class Metric(StrEnum):
    PRIME_RENT = "prime_rent"
    GRADE_A_RENT = "grade_a_rent"
    VACANCY_RATE = "vacancy_rate"
    AVAILABILITY = "availability"
    TAKE_UP = "take_up"
    UNDER_OFFER = "under_offer"
    UNDER_CONSTRUCTION = "under_construction"
    PRE_LET_SHARE = "pre_let_share"
    PRIME_YIELD = "prime_yield"
    INVESTMENT_VOLUME = "investment_volume"


class Submarket(StrEnum):
    CENTRAL_LONDON = "Central London"
    CITY_CORE = "City Core"
    CITY_FRINGE = "City Fringe"
    WEST_END_CORE = "West End Core"
    WEST_END = "West End"
    MIDTOWN = "Midtown"
    KINGS_CROSS_EUSTON = "King's Cross/Euston"
    SOUTHBANK = "Southbank"
    CANARY_WHARF_DOCKLANDS = "Canary Wharf/Docklands"


class Broker(StrEnum):
    JLL = "JLL"
    SAVILLS = "Savills"
    BNP_PARIBAS = "BNP Paribas"
    CUSHMAN_WAKEFIELD = "Cushman & Wakefield"
    CBRE = "CBRE"
    AVISON_YOUNG = "Avison Young"
    DELOITTE = "Deloitte"


class SignalType(StrEnum):
    DISTRESS = "distress"
    DEMAND = "demand"
    SUPPLY = "supply"
    PRICING = "pricing"
    POLICY = "policy"
    OTHER = "other"


class CardKind(StrEnum):
    """What a card is to Nan Fung, said first on screen for readers who are not property-finance specialists."""
    CHANCE_TO_BUY = "chance_to_buy"
    CHANCE_TO_LEASE = "chance_to_lease"
    RISK = "risk"
    WATCH = "watch"


class SourceKind(StrEnum):
    NEWS = "news"
    GAZETTE = "gazette"
    BOE = "boe"


class DbRole(StrEnum):
    OWNER = "owner"
    APP = "app"
    MCP = "mcp"


# The Postgres login behind each role: the owner migrates and seeds, the app runs the agent, the MCP server reads.
DB_ROLE_USERS = {DbRole.OWNER: "lom_owner", DbRole.APP: "lom_app", DbRole.MCP: "lom_mcp"}

# Every figure of a metric is stored in this unit, so values from different brokers compare without conversion.
METRIC_UNITS = {
    Metric.PRIME_RENT: "GBP psf pa",
    Metric.GRADE_A_RENT: "GBP psf pa",
    Metric.VACANCY_RATE: "%",
    Metric.AVAILABILITY: "m sq ft",
    Metric.TAKE_UP: "m sq ft",
    Metric.UNDER_OFFER: "m sq ft",
    Metric.UNDER_CONSTRUCTION: "m sq ft",
    Metric.PRE_LET_SHARE: "%",
    Metric.PRIME_YIELD: "%",
    Metric.INVESTMENT_VOLUME: "GBP bn",
}

# Knight Frank's reports prohibit use of their data on AI platforms (owner decision D1), so the fact store refuses it.
EXCLUDED_BROKERS = frozenset({"Knight Frank"})

# Bank of England series the agent may read: code -> plain name.
MACRO_SERIES = {"IUDBEDR": "Bank Rate", "IUDMNPY": "10-year gilt yield"}

# Tags on the synthetic fixture items that the scan evaluations look for or must ignore.
FIXTURE_ROLES = ("insolvency_lead", "ai_letting", "supply_gap", "decoy_residential", "decoy_retail", "injection")

# The office-data MCP server's tools. Kept here so the agent package can name them without importing the server.
MCP_TOOL_NAMES = frozenset({
    "get_coverage", "get_metric", "compare_brokers", "metric_change", "get_macro", "search_documents",
    "refresh_sources", "get_portfolio", "get_mandate", "match_to_mandate", "similar_past_decisions", "get_citation",
    "metric_trend",
})
