"""One validated model per seed-file row. Unknown fields are refused, and so is anything from Knight Frank."""
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from ..vocab import EXCLUDED_BROKERS, METRIC_UNITS, Broker, Metric, Submarket

PERIOD_PATTERN = r"^\d{4}(Q[1-4]|-(0[1-9]|1[0-2]))$"
SOURCE_KEY_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
LOCATOR_MAX_LENGTH = 60
EXCLUDED_BROKER_MESSAGE = "Knight Frank is excluded (owner decision D1)"
# Knight Frank's reports prohibit use of their data on AI platforms, so none of its documents may be a source.
EXCLUDED_SOURCE_HOST_MARK = "knightfrank."


def _refuse_excluded_broker(value):
    if isinstance(value, str) and value.strip() in EXCLUDED_BROKERS:
        raise ValueError(EXCLUDED_BROKER_MESSAGE)
    return value


class SeedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceRow(SeedModel):
    source_key: str = Field(pattern=SOURCE_KEY_PATTERN)
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    # Some sources (a company's property page) carry no date; blank stays blank rather than a made-up date.
    published: date | None
    licence_note: str = Field(min_length=1)

    @field_validator("published", mode="before")
    @classmethod
    def _blank_date_is_none(cls, value):
        return None if value == "" else value

    @field_validator("url")
    @classmethod
    def _not_an_excluded_source(cls, url: HttpUrl) -> HttpUrl:
        if EXCLUDED_SOURCE_HOST_MARK in (url.host or "").lower():
            raise ValueError("Knight Frank sources are excluded (owner decision D1)")
        return url


class BrokerMetricRow(SeedModel):
    broker: Broker
    metric: Metric
    submarket: Submarket
    broker_label: str = Field(min_length=1)
    period: str = Field(pattern=PERIOD_PATTERN)
    value: Decimal = Field(allow_inf_nan=False)
    unit: str
    measure_note: str = Field(min_length=1)
    source_key: str
    checked_at: date
    check_method: Literal["automatic", "manual"]
    locator: str | None = Field(default=None, max_length=LOCATOR_MAX_LENGTH)

    _broker_not_excluded = field_validator("broker", mode="before")(
        classmethod(lambda cls, v: _refuse_excluded_broker(v)))

    @field_validator("locator", mode="before")
    @classmethod
    def _blank_locator_is_none(cls, value):
        return None if value == "" else value

    @model_validator(mode="after")
    def _unit_matches_metric(self):
        expected = METRIC_UNITS[self.metric]
        if self.unit != expected:
            raise ValueError(f"unit must be {expected!r} for {self.metric.value}, got {self.unit!r}")
        return self


class DefinitionRow(SeedModel):
    broker: Broker
    metric: Metric
    definition: str = Field(min_length=1)
    source_key: str

    _broker_not_excluded = field_validator("broker", mode="before")(
        classmethod(lambda cls, v: _refuse_excluded_broker(v)))


class AreaMapRow(SeedModel):
    broker: Broker
    broker_label: str = Field(min_length=1)
    submarket: Submarket
    fit: Literal["exact", "approximate"]
    note: str = ""

    _broker_not_excluded = field_validator("broker", mode="before")(
        classmethod(lambda cls, v: _refuse_excluded_broker(v)))


class PortfolioAssetRow(SeedModel):
    name: str = Field(min_length=1)
    address: str = Field(min_length=1)
    submarket: Submarket
    size_sqft: int | None = Field(default=None, gt=0)
    status: str = Field(min_length=1)
    notes: str = ""
    source_keys: list[str] = Field(min_length=1)


class MandateDoc(SeedModel):
    label: Literal["illustrative"]
    target_submarkets: list[Submarket] = Field(min_length=1)
    size_band_sqft: tuple[int, int]
    strategies: list[str] = Field(min_length=1)
    signals_of_interest: list[str] = Field(min_length=1)
    statements: list[str] = Field(min_length=1)

    @field_validator("size_band_sqft")
    @classmethod
    def _band_is_ordered(cls, band: tuple[int, int]) -> tuple[int, int]:
        if not 0 < band[0] < band[1]:
            raise ValueError("size band must be two increasing positive numbers")
        return band

    @field_validator("statements")
    @classmethod
    def _statements_not_blank(cls, statements: list[str]) -> list[str]:
        if any(not s.strip() for s in statements):
            raise ValueError("statements must not be blank")
        return statements
