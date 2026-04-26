"""Sector-specific interpretation profiles for the analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SectorProfile:
    key: str
    label: str
    business_model: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    avoid_metrics: tuple[str, ...]
    valuation_caveat: str
    risk_caveat: str
    ai_prompt_hint: str

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_PROFILE = SectorProfile(
    key="default",
    label="General operating company",
    business_model="Operating company with mixed growth, profitability and balance-sheet drivers.",
    primary_metrics=("P/E", "P/S", "FCF yield", "balance-sheet quality"),
    secondary_metrics=("sales growth", "EPS growth", "analyst targets"),
    avoid_metrics=("single-metric verdicts",),
    valuation_caveat="Use a blended valuation and verify that sales, earnings and cash-flow all point in the same direction.",
    risk_caveat="The biggest risk is usually relying on one valuation method while the underlying data is incomplete.",
    ai_prompt_hint="Treat as a normal operating company; compare growth, valuation, cash generation and balance-sheet quality.",
)


SECTOR_PROFILES = {
    "financial": SectorProfile(
        key="financial",
        label="Bank / financial institution",
        business_model="Regulated financial institution where capital, credit quality and funding mix matter more than net cash.",
        primary_metrics=("P/E", "P/B when available", "ROE", "capital strength", "credit quality"),
        secondary_metrics=("deposit/funding stability", "dividend durability", "provision cycle"),
        avoid_metrics=("net cash", "FCF yield", "Altman Z", "operating-company DCF"),
        valuation_caveat="For banks and financials, do not treat net debt or FCF yield like an industrial company.",
        risk_caveat="Watch credit losses, capital ratios, funding pressure and rate sensitivity before judging the balance sheet.",
        ai_prompt_hint="Use bank-specific logic: capital, credit cycle, funding stability and P/E/P/B style valuation.",
    ),
    "energy": SectorProfile(
        key="energy",
        label="Energy / commodity producer",
        business_model="Commodity-exposed producer where earnings and cash-flow move with oil, gas or resource prices.",
        primary_metrics=("cycle-normalized P/E", "FCF yield", "net debt", "reserves/production", "commodity sensitivity"),
        secondary_metrics=("dividend/buyback durability", "hedging", "capex intensity"),
        avoid_metrics=("one-year growth extrapolation", "single-cycle P/E verdict"),
        valuation_caveat="For energy, a low P/E can be real but cyclical; normalize across the commodity cycle.",
        risk_caveat="The main risk is commodity price reversal, reserve quality, political exposure and capex discipline.",
        ai_prompt_hint="Explain commodity cyclicality, balance-sheet resilience, FCF durability and cycle-adjusted valuation.",
    ),
    "software": SectorProfile(
        key="software",
        label="Software / platform",
        business_model="Asset-light growth business where revenue growth, margins, retention and valuation multiples drive the thesis.",
        primary_metrics=("revenue growth", "FCF margin", "Rule of 40", "P/S", "gross margin"),
        secondary_metrics=("EPS inflection", "customer growth", "operating leverage"),
        avoid_metrics=("P/E-only verdict when earnings are early",),
        valuation_caveat="For software/platforms, P/S can be more informative than P/E when earnings are still scaling.",
        risk_caveat="Watch decelerating growth, multiple compression and whether growth converts into durable FCF.",
        ai_prompt_hint="Use growth-quality logic: revenue growth, margin expansion, Rule of 40, P/S and FCF conversion.",
    ),
    "industrial": SectorProfile(
        key="industrial",
        label="Industrial / aerospace / defense",
        business_model="Cyclical or project-driven industrial business where backlog, margins and FCF conversion matter.",
        primary_metrics=("P/E", "EV/EBITDA if available", "FCF conversion", "backlog/order trends", "debt"),
        secondary_metrics=("margin stability", "capex cycle", "contract visibility"),
        avoid_metrics=("pure P/S comparison",),
        valuation_caveat="For industrials, P/E and FCF conversion usually matter more than a pure sales multiple.",
        risk_caveat="Watch execution risk, cost inflation, backlog quality and leverage through the cycle.",
        ai_prompt_hint="Discuss execution, backlog, margin durability, FCF conversion and leverage.",
    ),
    "healthcare": SectorProfile(
        key="healthcare",
        label="Healthcare / pharma",
        business_model="Healthcare business influenced by pipeline, regulation, patents and reimbursement.",
        primary_metrics=("P/E", "sales growth", "pipeline/regulatory catalysts", "FCF durability"),
        secondary_metrics=("patent cliff risk", "R&D intensity", "balance-sheet flexibility"),
        avoid_metrics=("near-term sales growth alone",),
        valuation_caveat="For healthcare, valuation has to be read with pipeline and patent/regulatory risk.",
        risk_caveat="Watch trial outcomes, regulatory events, reimbursement pressure and concentration in key products.",
        ai_prompt_hint="Explain pipeline, regulatory risk, patent durability and earnings quality.",
    ),
}


def get_sector_profile(sector_name: str | None, benchmark_name: str | None = None) -> SectorProfile:
    """Return the most useful sector interpretation profile for a stock."""
    text = " ".join([str(sector_name or ""), str(benchmark_name or "")]).lower()

    if any(keyword in text for keyword in ("bank", "financial", "insurance", "capital markets", "asset management")):
        return SECTOR_PROFILES["financial"]
    if any(keyword in text for keyword in ("energy", "oil", "gas", "commodity", "mining", "metals", "gold", "copper")):
        return SECTOR_PROFILES["energy"]
    if any(keyword in text for keyword in ("software", "saas", "cloud", "consumer apps", "platform", "cybersecurity")):
        return SECTOR_PROFILES["software"]
    if any(keyword in text for keyword in ("aerospace", "defense", "industrial", "machinery", "transportation")):
        return SECTOR_PROFILES["industrial"]
    if any(keyword in text for keyword in ("healthcare", "pharma", "biotech", "medical")):
        return SECTOR_PROFILES["healthcare"]

    return DEFAULT_PROFILE


def sector_profile_summary(profile: SectorProfile | dict) -> str:
    """Compact sentence used in UI cards and export reports."""
    payload = profile.as_dict() if isinstance(profile, SectorProfile) else dict(profile or {})
    label = payload.get("label") or DEFAULT_PROFILE.label
    primary = ", ".join(payload.get("primary_metrics") or DEFAULT_PROFILE.primary_metrics[:3])
    return f"{label}: priorite a {primary}."
