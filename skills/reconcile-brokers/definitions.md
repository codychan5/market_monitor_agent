# How the brokers measure (crib sheet)

Each figure's own definition and measure note come back with get_metric and compare_brokers; this sheet explains the usual reasons the same metric differs between brokers. Knight Frank is not in the store: its reports prohibit use of its data on AI platforms.

## Rents
- **Headline prime rent**: the rent quoted for the best space in a submarket, before incentives. Brokers that publish it per submarket core (for example "City Core") report higher figures than wider-area averages.
- **Average achieved rent** (Savills reports average achieved prime and Grade A rents): what deals actually signed at in the period. It moves with the mix of deals, so one large letting can swing it.
- **Prime versus premium**: BNP Paribas Real Estate splits "prime" from a higher "premium" tier, so its prime figure can sit below another broker's.
- **Net effective rent**: headline rent spread over the lease after rent-free periods. Not stored here; a long rent-free period can make two equal headline rents very different in value.

## Space
- **Availability**: space on the market now plus space that will complete within a horizon. The horizon differs: CBRE counts marketed space that will be ready within 12 months, others use shorter windows. A longer horizon gives a higher figure.
- **Vacancy rate**: availability (or vacant space only, depending on the broker) as a share of total stock. Brokers' stock totals and boundaries differ, so rates differ by a point or more.
- **Take-up**: space let or sold for occupation in the period. Deal-size thresholds differ: BNP Paribas counts deals of 1,000 sq ft or more; Cushman & Wakefield's annual London Moves study counts deals over 5,000 sq ft. Some include pre-lets and lease renewals, some do not.
- **Under offer**: space where terms are agreed but not signed.

## Supply and investment
- **Under construction**: schemes on site; thresholds (for example 10,000 or 20,000 sq ft) and refurbishment rules differ. The pre-let share is the part already let or under offer.
- **Prime yield**: the yield on the best stock at a typical lot size; it differs with the lot size and the broker's view of comparable evidence.
- **Investment volume**: deals completed in the period; thresholds (for example £1m or more) and whether site purchases count differ.

## Areas
- Brokers draw submarket boundaries differently (for example the City versus the City Core, Docklands versus Canary Wharf). A row whose area_fit is "approximate" covers a wider or narrower area than the store's submarket.
