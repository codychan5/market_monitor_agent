"""Bank of England series (Bank Rate, 10-year gilt yield) from the IADB CSV download."""
import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from ..vocab import MACRO_SERIES

BOE_URL = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom={date_from}"
           "&Dateto=now&SeriesCodes={codes}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
BOE_DATE_FORMAT = "%d %b %Y"
BOE_PUBLISHER = "Bank of England"


@dataclass(frozen=True)
class MacroPoint:
    series: str
    day: date
    value: Decimal


class BoeParseError(Exception):
    """The download is not the expected DATE,<series>... CSV."""


def boe_url(date_from: date) -> str:
    return BOE_URL.format(date_from=date_from.strftime("%d/%b/%Y"), codes=",".join(MACRO_SERIES))


def parse_boe_csv(text: str) -> list[MacroPoint]:
    """One point per series per day; blank cells (no observation that day) are skipped."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or reader.fieldnames[0] != "DATE":
        raise BoeParseError("expected a CSV whose first column is DATE")
    points = []
    for row in reader:
        day = datetime.strptime(row["DATE"].strip(), BOE_DATE_FORMAT).date()
        for series in reader.fieldnames[1:]:
            cell = (row.get(series) or "").strip()
            if not cell:
                continue
            try:
                points.append(MacroPoint(series, day, Decimal(cell)))
            except InvalidOperation as e:
                raise BoeParseError(f"{series} on {day}: not a number") from e
    return points
