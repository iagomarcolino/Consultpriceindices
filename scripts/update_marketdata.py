import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
SYMBOLS = {
    # =========================
    # 🇺🇸 NEW YORK (NYSE)
    # =========================
    "^DJI": "Dow Jones Industrial Average (DJIA)",
    "^NYA": "NYSE Composite Index",

    # =========================
    # 🇺🇸 CHICAGO / NASDAQ
    # =========================
    "^IXIC": "Nasdaq Composite Index",
    "^NDX": "Nasdaq 100 (NDX)",

    # =========================
    # 🇨🇦 TORONTO
    # =========================
    "^GSPTSE": "S&P/TSX Composite Index",
    #"^TX60": "S&P/TSX 60 Index",   #não funciona
    # Toronto (TSX 60) - troca ^TX60
    "TX60.TS": "S&P/TSX 60 Index",  # índice no Yahoo
    # ou, se preferir proxy via ETF:
    # "XIU.TO": "iShares S&P/TSX 60 Index ETF"

    # =========================
    # 🇬🇧 LONDON
    # =========================
    "^FTSE": "FTSE 100",
    "^FTMC": "FTSE 250",

    # =========================
    # 🇪🇺 EURONEXT
    # =========================
    "^FCHI": "CAC 40 (France)",
    "^AEX": "AEX (Netherlands)",
    "^BFX": "BEL 20 (Belgium)",

    # =========================
    # 🇩🇪 FRANKFURT
    # =========================
    "^GDAXI": "DAX 40 (Germany)",
    "^MDAXI": "MDAX (Germany Mid Caps)",

    # =========================
    # 🇨🇭 ZURICH
    # =========================
    "^SSMI": "SMI - Swiss Market Index",
    "^SSHI": "SPI - Swiss Performance Index",

    # =========================
    # 🇮🇳 INDIA
    # =========================
    "^BSESN": "SENSEX (India)",
    "^NSEI": "NIFTY 50 (India)",

    # =========================
    # 🇧🇷 BRAZIL - B3
    # =========================
    "^BVSP": "Ibovespa (IBOV)",
    #"^IBX100": "IBrX 100", #não funciona
    "^IBX50": "IBrX 50",
    # Brasil (IBrX 100) - troca ^IBX100
    "BRAX11.SA": "iShares IBrX-Índice Brasil (IBrX-100) ETF (proxy do IBrX 100)",

    # =========================
    # 🇯🇵 JAPAN - TOKYO
    # =========================
    "^N225": "Nikkei 225",
    #"^TOPX": "TOPIX",  #não funciona
    "1306.T": "NEXT FUNDS TOPIX ETF (proxy do TOPIX)",
    # ou:
    # "1475.T": "iShares Core TOPIX ETF (proxy do TOPIX)",
    
    # =========================
    # 🇰🇷 SOUTH KOREA - SEOUL
    # =========================
    "^KS11": "KOSPI (South Korea)",
    "^KQ11": "KOSDAQ (South Korea)",

    # =========================
    # 🇨🇳 CHINA - SHANGHAI / SHENZHEN
    # =========================
    "000001.SS": "SSE Composite Index (Shanghai)",
    "000300.SS": "CSI 300 (Shanghai + Shenzhen)",

    "399001.SZ": "SZSE Component Index (Shenzhen)",
    "399006.SZ": "ChiNext Index (Shenzhen)",

    # =========================
    # 🇭🇰 HONG KONG
    # =========================
    "^HSI": "Hang Seng Index (HK50)",

    # =========================
    # 🇦🇺 AUSTRALIA - SYDNEY
    # =========================
    "^AXJO": "S&P/ASX 200",

    # =========================
    # 🇸🇬 SINGAPORE
    # =========================
    "^STI": "Straits Times Index (Singapore)",
}

LOOKBACK = "400d"
INTERVAL = "1d"
TRADING_DAYS = 252

# janelas (em pregões) para vol anualizada por janela
WINDOWS = {
    "weekly": 5,
    "monthly": 21,
    "quarterly": 63,
    "semiannual": 126,
}

OUT_JSON = "data/marketdata.json"
OUT_CSV = None  # ex.: "data/marketdata.csv"


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _append_nulls(results, batch):
    """Se um batch falhar, adiciona linhas com null para não quebrar o pipeline."""
    for sym in batch:
        company_name = SYMBOLS.get(sym, sym)
        results.append(
            {
                "symbol": sym,
                "name": company_name,
                "price": None,
                "vol_annual": None,
                "vol_semiannual": None,
                "vol_quarterly": None,
                "vol_monthly": None,
                "vol_weekly": None,
            }
        )


def _extract_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai a matriz de fechamentos (Close) para 1 ou vários tickers."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0, drop_level=True)
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close

    return pd.DataFrame()


def _ann_vol_from_logret_window(logret: pd.DataFrame, window: int) -> pd.Series:
    """
    Vol anualizada estimada usando apenas os últimos 'window' retornos diários.
    Retorna Series indexada pelos tickers.
    """
    if logret is None or logret.empty:
        return pd.Series(dtype="float64")

    tail = logret.tail(window)

    # Precisa de pelo menos 2 observações para std com ddof=1
    if len(tail.index) < 2:
        return pd.Series(index=logret.columns, dtype="float64")

    return tail.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    results = []

    for batch in chunked(list(SYMBOLS.keys()), 100):
        try:
            df = yf.download(
                tickers=batch,
                period=LOOKBACK,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:
            print(f"[WARN] Batch falhou (exception): {e}")
            _append_nulls(results, batch)
            continue

        close = _extract_close_df(df)

        if close.empty or len(close.index) == 0:
            print("[WARN] Batch retornou vazio/sem Close. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        for sym in batch:
            if sym not in close.columns:
                close[sym] = np.nan

        close = close[batch]

        close_ffill = close.ffill()
        if close_ffill.empty or len(close_ffill.index) == 0:
            print("[WARN] Close após ffill ficou vazio. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        last_price = close_ffill.iloc[-1]

        # Log-retornos diários
        logret = np.log(close / close.shift(1))

        # Vol anualizada usando todo o período
        vol_annual = logret.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

        # Vol anualizada por janelas
        vol_weekly = _ann_vol_from_logret_window(logret, WINDOWS["weekly"])
        vol_monthly = _ann_vol_from_logret_window(logret, WINDOWS["monthly"])
        vol_quarterly = _ann_vol_from_logret_window(logret, WINDOWS["quarterly"])
        vol_semiannual = _ann_vol_from_logret_window(logret, WINDOWS["semiannual"])

        for sym in batch:
            p = last_price.get(sym, np.nan)

            vA = vol_annual.get(sym, np.nan)
            vW = vol_weekly.get(sym, np.nan)
            vM = vol_monthly.get(sym, np.nan)
            vQ = vol_quarterly.get(sym, np.nan)
            vS = vol_semiannual.get(sym, np.nan)

            company_name = SYMBOLS.get(sym, sym)

            results.append(
                {
                    "symbol": sym,
                    "name": company_name,
                    "price": None if pd.isna(p) else float(round(float(p), 6)),

                    # anualizada (com base na janela indicada)
                    "vol_annual": None if pd.isna(vA) else float(round(float(vA), 8)),
                    "vol_semiannual": None if pd.isna(vS) else float(round(float(vS), 8)),
                    "vol_quarterly": None if pd.isna(vQ) else float(round(float(vQ), 8)),
                    "vol_monthly": None if pd.isna(vM) else float(round(float(vM), 8)),
                    "vol_weekly": None if pd.isna(vW) else float(round(float(vW), 8)),
                }
            )

    payload = {
        "generated_at_utc": now,
        "source": "yfinance",
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "trading_days": TRADING_DAYS,
        "count": len(results),
        "data": results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if OUT_CSV:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8")

    ok_prices = sum(1 for r in results if r["price"] is not None)
    ok_volA = sum(1 for r in results if r["vol_annual"] is not None)
    ok_volW = sum(1 for r in results if r["vol_weekly"] is not None)
    ok_volM = sum(1 for r in results if r["vol_monthly"] is not None)
    ok_volQ = sum(1 for r in results if r["vol_quarterly"] is not None)
    ok_volS = sum(1 for r in results if r["vol_semiannual"] is not None)

    print(f"OK: atualizado {OUT_JSON} com {len(results)} tickers.")
    print(f"   Preços OK: {ok_prices}")
    print(f"   Vols OK: anual={ok_volA} | semanal={ok_volW} | mensal={ok_volM} | trimestral={ok_volQ} | semestral={ok_volS}")


if __name__ == "__main__":
    main()
