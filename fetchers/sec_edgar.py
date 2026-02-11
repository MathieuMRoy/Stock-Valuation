"""
SEC EDGAR Fetcher - Official Data from data.sec.gov (JSON XBRL)
"""
import requests
import pandas as pd
import streamlit as st

# SEC Requires a proper User-Agent with email
HEADERS = {
    "User-Agent": "ValuationMasterPro/2.0 (nicolas.perron.pro@example.com)"
}

@st.cache_data(ttl=3600*24*7) # Cache CIK mapping for a week
def get_cik_mapping():
    """Download Ticker -> CIK mapping from SEC"""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            mapping = {}
            for key, val in data.items():
                mapping[val["ticker"].upper()] = str(val["cik_str"]).zfill(10)
            return mapping
    except:
        return {}
    return {}

@st.cache_data(ttl=3600*24)
def get_sec_financials(ticker: str) -> dict:
    """
    Fetch financial data directly from SEC EDGAR API (companyfacts).
    Parses XBRL tags to reconstruct income statement history.
    """
    mapping = get_cik_mapping()
    cik = mapping.get(ticker.upper())
    
    out = {"inc_a": pd.DataFrame(), "inc_q": pd.DataFrame(), "error": None}
    
    if not cik:
        out["error"] = f"CIK lookup failed for {ticker}. SEC data unavailable."
        return out
        
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            out["error"] = f"SEC API Error: {r.status_code}"
            return out
            
        data = r.json()
        facts = data.get("facts", {}).get("us-gaap", {})
        
        # Define Tags to look for (Handling taxonomy changes)
        # We process them in order of preference or sum them?
        # Ideally we pick the one that has data for the year.
        
        metrics_map = {
            "Total Revenue": ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueGoodsNet"],
            "Net Income": ["NetIncomeLoss", "ProfitLoss"],
            "EPS": ["EarningsPerShareBasic"],
            "Operating Income": ["OperatingIncomeLoss"],
            "Gross Profit": ["GrossProfit"],
            "Cash From Operations": ["NetCashProvidedByUsedInOperatingActivities"],
            "CapEx": ["PaymentsToAcquirePropertyPlantAndEquipment"],
            "Depreciation & Amort": ["DepreciationDepletionAndAmortization", "Depreciation", "AmortizationOfIntangibleAssets"],
            "Dividends Paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"]
        }
        
        # Prepare storage
        pivoted_a = {} # Year -> {Metric: Val}
        pivoted_q = {} # Date -> {Metric: Val}
        
        for metric_name, tags in metrics_map.items():
            combined_points = []
            for tag in tags:
                if tag in facts:
                    units_map = facts[tag].get("units", {})
                    # Try USD, then USD/shares
                    units = units_map.get("USD", [])
                    if not units:
                        units = units_map.get("USD/shares", [])
                    
                    combined_points.extend(units)
            
            if not combined_points:
                continue
                
            df = pd.DataFrame(combined_points)
            if df.empty: continue
            
            # --- Annual (10-K) ---
            df_a = df[df["form"] == "10-K"].copy()
            if not df_a.empty:
                df_a = df_a.sort_values("filed")
                # Deduplicate by FY
                df_a = df_a.drop_duplicates(subset=["fy"], keep="last")
                
                for _, row in df_a.iterrows():
                    fy = str(int(row["fy"])) if pd.notna(row["fy"]) else None
                    if fy:
                        if fy not in pivoted_a: pivoted_a[fy] = {}
                        pivoted_a[fy][metric_name] = row["val"]

            # --- Quarterly (10-Q) ---
            df_q = df[df["form"] == "10-Q"].copy()
            if not df_q.empty:
                df_q = df_q.sort_values("filed")
                # Deduplicate by END date
                df_q = df_q.drop_duplicates(subset=["end"], keep="last")
                
                for _, row in df_q.iterrows():
                    date = row["end"]
                    if date not in pivoted_q: pivoted_q[date] = {}
                    pivoted_q[date][metric_name] = row["val"]
        
        # Helper: Calculate derived metrics
        def calc_derived(pivoted_data):
            for date_key, metrics in pivoted_data.items():
                # FCF
                cfo = metrics.get("Cash From Operations", 0)
                capex = metrics.get("CapEx", 0)
                if cfo or capex:
                    metrics["Free Cash Flow"] = cfo - abs(capex)
                
                # EBITDA
                op_inc = metrics.get("Operating Income", 0)
                da = metrics.get("Depreciation & Amort", 0)
                if op_inc or da:
                    metrics["EBITDA"] = op_inc + da
                    
        calc_derived(pivoted_a)
        calc_derived(pivoted_q)

        # Construct Final DataFrames
        if pivoted_a:
            df_final_a = pd.DataFrame.from_dict(pivoted_a, orient='index')
            df_final_a.index.name = "Year"
            df_final_a = df_final_a.T # Index=Metric, Cols=Year
            # Filter Annual >= 2018 (Optional logic here or in UI, doing broad fetch here is safer)
            out["inc_a"] = df_final_a
 
        if pivoted_q:
            df_final_q = pd.DataFrame.from_dict(pivoted_q, orient='index')
            df_final_q.index.name = "Date"
            df_final_q = df_final_q.T # Index=Metric, Cols=Date
            out["inc_q"] = df_final_q
            
    except Exception as e:
        out["error"] = f"SEC Processing Error: {str(e)}"
        
    return out

def get_available_metrics_sec(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    return list(df.index)
