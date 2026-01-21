import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
SYMBOLS = {
    "ABEV3.SA": "Ambev",
    "ALOS3.SA": "Allos",
    "ASAI3.SA": "Assaí",
    "AURE3.SA": "Auren Energia",
    "AXIA3.SA": "Axia (Verificar Ticker)", # Provável erro ou ticker específico
    "AXIA6.SA": "Axia (Verificar Ticker)",
    "AXIA7.SA": "Axia (Verificar Ticker)",
    "AZZA3.SA": "Azzas 2154 (Arezzo+Soma)",
    "B3SA3.SA": "B3",
    "BBAS3.SA": "Banco do Brasil",
    "BBDC3.SA": "Bradesco (ON)",
    "BBDC4.SA": "Bradesco (PN)",
    "BBSE3.SA": "BB Seguridade",
    "BEEF3.SA": "Minerva",
    "BPAC11.SA": "BTG Pactual",
    "BRAP4.SA": "Bradespar",
    "BRAV3.SA": "Brava Energia (3R + Enauta)",
    "BRKM5.SA": "Braskem",
    "CEAB3.SA": "C&A",
    "CMIG4.SA": "Cemig",
    "CMIN3.SA": "CSN Mineração",
    "COGN3.SA": "Cogna",
    "CPFE3.SA": "CPFL Energia",
    "CPLE3.SA": "Copel",
    "CSAN3.SA": "Cosan",
    "CSMG3.SA": "Copasa",
    "CSNA3.SA": "CSN Siderúrgica",
    "CURY3.SA": "Cury Construtora",
    "CXSE3.SA": "Caixa Seguridade",
    "CYRE3.SA": "Cyrela",
    "CYRE4.SA": "Cyrela (PN)", # Pouco líquida
    "DIRR3.SA": "Direcional",
    "EGIE3.SA": "Engie Brasil",
    "EMBJ3.SA": "Embraer (Verificar: EMBR3)",
    "ENEV3.SA": "Eneva",
    "ENGI11.SA": "Energisa",
    "EQTL3.SA": "Equatorial",
    "FLRY3.SA": "Fleury",
    "GGBR4.SA": "Gerdau",
    "GOAU4.SA": "Metalúrgica Gerdau",
    "HAPV3.SA": "Hapvida",
    "HYPE3.SA": "Hypera Pharma",
    "IGTI11.SA": "Iguatemi",
    "IRBR3.SA": "IRB Re",
    "ISAE4.SA": "ISA CTEEP (TRPL4)",
    "ITSA4.SA": "Itaúsa",
    "ITUB4.SA": "Itaú Unibanco",
    "KLBN11.SA": "Klabin",
    "LREN3.SA": "Lojas Renner",
    "MBRF3.SA": "Marfrig (Verificar: MRFG3)",
    "MGLU3.SA": "Magazine Luiza",
    "MOTV3.SA": "Movida (Verificar: MOVI3)",
    "MRVE3.SA": "MRV",
    "MULT3.SA": "Multiplan",
    "NATU3.SA": "Natura",
    "PCAR3.SA": "Pão de Açúcar",
    "PETR3.SA": "Petrobras (ON)",
    "PETR4.SA": "Petrobras (PN)",
    "POMO4.SA": "Marcopolo",
    "PRIO3.SA": "Prio (PetroRio)",
    "PSSA3.SA": "Porto Seguro",
    "RADL3.SA": "Raia Drogasil",
    "RAIL3.SA": "Rumo",
    "RAIZ4.SA": "Raízen",
    "RDOR3.SA": "Rede D'Or",
    "RECV3.SA": "PetroReconcavo",
    "RENT3.SA": "Localiza",
    "RENT4.SA": "Localiza (PN - Antiga)",
    "SANB11.SA": "Santander Brasil",
    "SBSP3.SA": "Sabesp",
    "SLCE3.SA": "SLC Agrícola",
    "SMFT3.SA": "Smart Fit",
    "SUZB3.SA": "Suzano",
    "TAEE11.SA": "Taesa",
    "TIMS3.SA": "TIM",
    "TOTS3.SA": "Totvs",
    "UGPA3.SA": "Ultrapar",
    "USIM5.SA": "Usiminas",
    "VALE3.SA": "Vale",
    "VAMO3.SA": "Vamos",
    "VBBR3.SA": "Vibra (BR Distribuidora)",
    "VIVA3.SA": "Vivara",
    "VIVT3.SA": "Vivo",
    "WEGE3.SA": "WEG",
    "YDUQ3.SA": "Yduqs (Estácio)",
}

LOOKBACK = "400d"
INTERVAL = "1d"
TRADING_DAYS = 252

OUT_JSON = "data/marketdata.json"
OUT_CSV = None  # ex.: "data/marketdata.csv"


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _append_nulls(results, batch):
    """Se um batch falhar, adiciona linhas com null para não quebrar o pipeline."""
    for sym in batch:
        # Busca o nome no dicionário SYMBOLS
        company_name = SYMBOLS.get(sym, sym) 
        
        results.append(
            {
                "symbol": sym,
                "name": company_name, 
                "price": None,
                "vol_annual": None,
            }
        )


def _extract_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai a matriz de fechamentos (Close) no formato:
      index = datas
      colunas = tickers
    Suporta o formato que o yfinance devolve para 1 ticker ou vários tickers.
    """
    # Caso o df venha vazio
    if df is None or df.empty:
        return pd.DataFrame()

    # Caso comum com group_by="column": df["Close"] funciona se existir
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close

    # Caso alternativo: MultiIndex nas colunas (ex.: ('Close', 'PETR4.SA'))
    if isinstance(df.columns, pd.MultiIndex):
        # tenta achar o nível "Close"
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0, drop_level=True)
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close

    # Se não conseguiu extrair
    return pd.DataFrame()


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

        # Se não veio nada de Close / veio vazio, não quebra: registra null e segue
        if close.empty or len(close.index) == 0:
            print("[WARN] Batch retornou vazio/sem Close. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        # Garante que todas as colunas estejam presentes (se alguns tickers falharam)
        # cria colunas faltantes com NaN
        for sym in batch:
            if sym not in close.columns:
                close[sym] = np.nan

        # Ordena colunas para consistência (opcional)
        close = close[batch]

        # Último preço conhecido por ticker
        close_ffill = close.ffill()
        # Se por algum motivo ainda não tiver linha depois do ffill, evita iloc[-1]
        if close_ffill.empty or len(close_ffill.index) == 0:
            print("[WARN] Close após ffill ficou vazio. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        last_price = close_ffill.iloc[-1]

        # Vol anualizada (log-retornos)
        logret = np.log(close / close.shift(1))
        vol = logret.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

        for sym in batch:
            p = last_price.get(sym, np.nan)
            v = vol.get(sym, np.nan)
            
            # Pega do cache, ou usa o ticker se não achar
            company_name = SYMBOLS.get(sym, sym)

            results.append(
                {
                    "symbol": sym,
                    "name": company_name, 
                    "price": None if pd.isna(p) else float(round(float(p), 6)),
                    "vol_annual": None if pd.isna(v) else float(round(float(v), 8)),
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
    ok_vols = sum(1 for r in results if r["vol_annual"] is not None)
    print(f"OK: atualizado {OUT_JSON} com {len(results)} tickers.")
    print(f"   Preços OK: {ok_prices} | Vols OK: {ok_vols}")


if __name__ == "__main__":
    main()
