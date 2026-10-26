# Free and public data sources for a London office PoC (checked from Hong Kong)

Research notes gathered on 2026-10-23 by a research sub-agent using web search; figures carry their source and date inline, and items marked unverified were not confirmed against a primary source.

I checked everything on 23 Oct 2026 from a Hong Kong connection, the same region as your reviewers. Every block I hit (CBRE, Colliers, Cluttons, Helical, CoStar, React News, tfl.gov.uk pages) came from bot protection, not geo-blocking.

| Source | What it gives | Access method | Key / registration | Licence / terms | Verified working? |
|---|---|---|---|---|---|
| BoE IADB | Bank Rate `IUDBEDR`, SONIA `IUDSOIA`, gilt par yields 5/10/20y `IUDSNPY`/`IUDMNPY`/`IUDLNPY`, GBP/USD `XUDLUSS`, GBP/HKD `XUDLHDS` | CSV: `https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2020&Dateto=now&SeriesCodes=IUDBEDR,XUDLHDS&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N` | No | OGL, but "selected exchange rate" series are excluded (check the FX ones) | Yes (gilts arrive a day late) |
| ONS | CPI `D7G7`/MM23, GDP q/q `IHYQ`/QNA, monthly GDP `ECY2`/MGDP, `MGSX`, `LF24`, `AP2Y` (LMS) | JSON, e.g. `ons.gov.uk/economy/inflationandpriceindices/timeseries/d7g7/mm23/data`. The old `api.ons.gov.uk` was retired 25 Nov 2024; `api.beta.ons.gov.uk/v1` only serves filterable datasets | No | OGL | Yes |
| Nomis | BRES by borough (annual, latest 2024); Workforce Jobs `NM_130_1` (London region only); claimant count `NM_162_1` | `nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv?geography=E09000001,E09000033,E09000030,E09000007,E09000019&date=latest&industry=37748736&employment_status=4&measures=20100` | Optional uid (anonymous: 25k cells per request) | OGL | Yes |
| Planning London Datahub | All London planning applications: office floorspace, start/completion dates | Undocumented: POST `planningdata.london.gov.uk/api-guest/applications/_search`, header `X-API-AllowRequest: be2rmRnt&` | No (public guest token) | Unofficial | Yes |
| London Datastore | GLA datasets | `data.london.gov.uk/api/action/package_show`; full catalogue at `/api/v2/datasets/export.json` (13.5 MB); search parameter is ignored | No | Mostly OGL | Partly |
| EPC (non-domestic) | Ratings, for MEES/ESG risk | REST + bulk CSV at `get-energy-performance-data.communities.gov.uk` (old opendatacommunities URL redirects there); 6,000 req/5 min | GOV.UK One Login token | OGL except address fields (energy-performance purposes only) | Not checked (docs only) |
| VOA | Rating lists; floorspace statistics | `voaratinglists.blob.core.windows.net/html/rlidata.htm`; "NDR stock of properties" on gov.uk | Click-through | Rating lists are licensed for business-rates purposes only, so **unsuitable**. Stock statistics are OGL | Pages only |
| HMLR CCOD/OCOD | Company-owned titles, company number, price paid; monthly | Bulk files + API, `use-land-property-data.service.gov.uk` | Free account, licence and API key | Free. Addresses limited to non-commercial or land-management use unless Royal Mail permits; no redistribution | Pages only |
| Companies House | Charges, insolvency, filings; streams `/charges`, `/insolvency-cases` | `api.company-information.service.gov.uk`, `stream.companieshouse.gov.uk` | Free key; 600 req/5 min | Free; officer data is personal data | Yes (key required) |
| The Gazette | Administrators `2410`, administrative receivers `2421`, winding-up petitions `2450`, liquidators `2432`/`2443` | Atom/JSON: `thegazette.co.uk/all-notices/notice/data.feed?noticetypes=2410` | No | OGL, excluding personal data | Yes |
| TfL | Daily entry/exit taps per station (to 17 Oct 2026); live crowding | `crowding.data.tfl.gov.uk/Network%20Demand/StationFootfall_2025_2026%20.csv` (11 MB); `api.tfl.gov.uk/crowding/940GZZLUBNK/Live` | Optional key (raises limit from 50 to 500 req/min) | Commercial use allowed; credit "Powered by TfL Open Data" | Yes |
| DfT transport use | Daily tube use as % of pre-COVID | CSV via `gov.uk/api/content/government/statistics/daily-domestic-transport-use-by-mode` | No | OGL | Yes |
| News RSS | Headlines | `cityam.com/category/property/feed/`, `propertyweek.com/news/feed` (section feeds stale), `bisnow.com/rss-feed/london`, `propertyindustryeye.com/feed/`, `feeds.bbci.co.uk/news/business/rss.xml`, `ft.com/property?format=rss`, `constructionenquirer.com/feed/` | No | Publisher copyright | Yes. No for EG (expired certificate/403), React News (captcha), CoStar (403) |
| GDELT DOC 2.0 | Article search | `api.gdeltproject.org/api/v2/doc/doc?query=…&mode=artlist&format=json` | No | Free with citation; about 1 req/5 s; noisy | Yes |
| Google News RSS | Search feed | `news.google.com/rss/search?q=…&hl=en-GB&gl=GB&ceid=GB:en` | No | Feed text: "personal, non-commercial use… Any other use… expressly prohibited" | Yes |
| NewsAPI | Search | REST | Key | Free tier: 100 req/day, 24 h delay, 1 month back, development only; paid from $449/mo | Yes (key required) |
| FCA NSM | Full RNS text for all six REITs, about 12 min after release | Undocumented: POST `api.data.fca.org.uk/search?index=nsm-search` | No | FCA terms not confirmed | Yes |
| REIT IR sites | Results: EPRA vacancy, ERV growth, leasing vs ERV | PDFs (e.g. `content.landsec.com`); Derwent RSS `derwentlondon.com/feeds/news.rss` | No | © | Yes; helical.co.uk blocked (403) |
| JLL Market Dynamics (quarterly) | Vacancy, submarket rents, pipeline, pre-let % | Predictable PDF: `jll.com/content/dam/jllcom/en/uk/documents/reports/research-reports/market-dynamics-office/q3/26-insights-central-london-office-q3-2026.pdf` | No | robots.txt disallows this path | Yes |
| Knight Frank Spotlight (quarterly) | Take-up, vacancy, submarkets | `knightfrank.co.uk/site-assets/research/report-pdfs/london-office-spotlight/spotlight-q2-2026.pdf` (file names vary) | No | **"Use of our data on AI platforms… or machine-learning purposes is strictly prohibited"** | Yes |
| Savills (monthly Market Watch; quarterly PDF) | Take-up, vacancy, rents | `pdf.euro.savills.co.uk/uk/office-reports/central-london-q2-2026-report.pdf` | No | No reproduction without permission; robots.txt allows | Yes |
| Cushman MarketBeat (quarterly) | Submarket table | `assets.cushmanwakefield.com/-/media/cw/marketbeat-pdfs/2026/q2/emea/central-london-office-marketbeat-q2-2026-low-res.pdf` | No | © | Yes from HK; 403 to a US-hosted fetcher |
| BNP PRE (quarterly) | Take-up, availability | `realestate.bnpparibas.co.uk/sites/default/files/2026-08/Q22026CentralLondonMarketUpdate.pdf` (path includes upload month) | No | No reproduction without consent | Yes |
| Deloitte Crane Survey (latest Apr 2026; used to be twice a year) | Starts, completions, refurbishment share | PDF from the hub on deloitte.com | No | © | Yes |
| Carter Jonas (quarterly + Net Effective Rents Monitor); Avison Young (quarterly) | Rents, incentives; charts | Carter Jonas: PDFs at unpredictable CDN URLs. Avison Young: web charts only, no PDF | No | © | Yes |
| CBRE, Colliers, Cluttons | Quarterly figures | Cloudflare challenge, including CBRE's PDFs | n/a | n/a | No (blocked) |

None of the broker PDFs I could reach sat behind a sign-up form. Only Cushman, JLL and Savills use quarter-based URLs; for the rest, find the link on the listing page each time.

**Top recommendations**
1. **Macro:** BoE + ONS JSON + Nomis. No keys needed, and all worked.
2. **Supply pipeline:** the Planning London Datahub API for office floorspace with start/completion dates, plus Deloitte totals and the Construction Enquirer feed.
3. **Distress signals:** Gazette feeds + Companies House charges/insolvency, matched to owners via CCOD company numbers (mind the address clause).
4. **Office attendance:** TfL taps at Bank, Liverpool Street, Canary Wharf, Oxford Circus and Tottenham Court Road, Tue–Thu vs 2019. Use the DfT series as a cross-check.
5. **REITs:** poll the FCA NSM and pull EPRA vacancy and ERV from results.
6. **Broker data, three options:**
   - Auto-fetch Savills/Cushman/BNP/Deloitte and keep only extracted figures with citations. Fully automated, some licence risk.
   - Have the analyst upload the PDFs. Lowest risk, but manual.
   - Use news headlines only. Lowest fidelity.
   - In all cases, keep Knight Frank and JLL out of automated ingestion.
7. **Keyed sources:** make Companies House, EPC and HMLR optional, read keys from env vars and skip cleanly without them. Reviewers would need their own accounts.

**Main gaps (only paid data covers these)**
- Deal-level comps: net effective rents, incentives, lease expiries, tenants (CoStar, EGi).
- Building-level availability, and Grade A/B vacancy history by submarket.
- Investment comps, yield history, MSCI return indices.
- Scheme-level pipeline with pre-let % (the Datahub lags).
- London-specific occupancy data (Remit only publishes UK-wide press releases).
- Consistent definitions: brokers' Q2 2026 Central London take-up figures range from 2.4m to 3.1m sq ft. Store source and definition, and never blend them.

**Flags**
- Whether summarising broker reports with an LLM is allowed for commercial buy-side use is a legal question. The UK text-and-data-mining exception (s29A CDPA) covers only non-commercial research, so this needs compliance sign-off rather than an assumption.
- I wrote no project files. The web-fetch tool automatically saved copies of the JLL and Knight Frank PDFs (about 1.6 MB) in `/Users/cody/.claude/projects/-Users-cody-Desktop-Custom-Projects-market-monitor-agent/75c1e296-cd4a-4a41-924f-b3d27f3d02e0/tool-results/`.
