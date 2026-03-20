import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybit.unified_trading import HTTP
from tqdm import tqdm

API_KEY    = ""
API_SECRET = ""


session = HTTP(demo=True, api_key=API_KEY, api_secret=API_SECRET)

chosen_portfolio = None
chosen_budget    = None
filters_cache    = {}
buy_spent        = {}


def get_top_pairs(n):
    tickers = session.get_tickers(category="spot")["result"]["list"]

    pairs = []
    for t in tickers:
        if t["symbol"].endswith("USDT"):
            vol = float(t.get("turnover24h") or 0)
            pairs.append((t["symbol"], vol))

    # corting
    pairs.sort(key=lambda x: x[1], reverse=True)

    top = []
    for symbol, vol in pairs[:n]:
        top.append(symbol)

    # delete stables and banned coins
    stables = ["USDCUSDT", "USDEUSDT", "FDUSDUSDT", "XMRUSDT", "ZCASH", "RLUSDUSDT"]
    for s in stables:
        if s in top:
            top.remove(s)

    return top


# getting data
def get_klines(pairs, days, interval):
    now   = int(time.time() * 1000)
    start = now - days * 24 * 60 * 60 * 1000

    result = {}
    for symbol in tqdm(pairs):
        try:
            r = session.get_kline(category="spot", symbol=symbol,
                                  interval=interval, start=start, end=now, limit=1000)
            result[symbol] = r
            time.sleep(0.12)
        except:
            pass

    return result


# creating the dataframe
def build_dataframe(klines_data):
    all_series = []

    for symbol, r in klines_data.items():
        candles = r["result"]["list"]
        prices  = {}
        for c in candles:
            prices[c[0]] = float(c[4])  # c[4] = цена закрытия
        s = pd.Series(prices, name=symbol)
        all_series.append(s)

    df = pd.concat(all_series, axis=1)
    df = df.dropna()
    df = df.sort_index()
    return df


def build_frontier(df, n_points=50):
    norm    = df / df.iloc[-1]
    returns = norm.pct_change().iloc[1:] * 100

    mu  = np.array(returns.mean())
    cov = np.array(returns.cov())
    inv = np.linalg.inv(cov)

    ones = np.ones(len(mu))

    A = float(mu @ inv @ mu)
    B = float(ones @ inv @ mu)
    C = float(ones @ inv @ ones)
    D = A * C - B ** 2

    portfolios     = []
    target_returns = np.linspace(float(mu.min()), float(mu.max()), n_points)

    for i in tqdm(range(len(target_returns))):
        target = target_returns[i]
        try:
            la1 = (C * target - B) / D
            la2 = (A - B * target) / D
            w   = la1 * (inv @ mu) + la2 * (inv @ ones)
            w   = w / np.sum(w)

            ret    = target
            var    = (C * target**2 - 2 * B * target + A) / D
            stderr = float(np.sqrt(var))

            weights = {}
            for j in range(len(df.columns)):
                if abs(w[j]) > 0.01:
                    weights[df.columns[j]] = w[j]

            portfolios.append({
                "n":       i + 1,
                "ret":     ret,
                "var":     var,
                "stderr":  stderr,
                "weights": weights,
            })
        except:
            pass

    return portfolios


def show_plot(portfolios):
    variances = []
    returns   = []
    for p in portfolios:
        variances.append(p["var"])
        returns.append(p["ret"])

    plt.figure(figsize=(6, 4))
    plt.scatter(variances, returns, color="red", s=30)

    for p in portfolios:
        plt.annotate(str(p["n"]), (p["var"], p["ret"]),
                     textcoords="offset points", xytext=(-10, 0),
                     fontsize=5, ha="center")

    plt.xlabel("Variance")
    plt.ylabel("Expected return")
    plt.title("Efficient Frontier")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# get limits
def get_filters(symbol):
    if symbol in filters_cache:
        return filters_cache[symbol]

    info = session.get_instruments_info(category="spot", symbol=symbol)
    lot  = info["result"]["list"][0]["lotSizeFilter"]

    step     = float(lot.get("basePrecision", "0.000001"))
    step_str = f"{step:.10f}".rstrip("0")
    decimals = len(step_str.split(".")[-1]) if "." in step_str else 0

    filters_cache[symbol] = {
        "step":       step,
        "decimals":   decimals,
        "max_qty":    float(lot.get("maxOrderQty", "1e18")),
        "min_amount": float(lot.get("minOrderAmt", "0")),
    }
    return filters_cache[symbol]


def buy_coin(symbol, usdt_amount):
    f = get_filters(symbol)

    if f["min_amount"] > 0 and usdt_amount < f["min_amount"]:
        print(f"  [skip] {symbol}: {usdt_amount:.2f} USDT < minimum {f['min_amount']:.2f}")
        return

    r    = session.place_order(category="spot", symbol=symbol, side="Buy",
                               orderType="Market", marketUnit="quoteCoin",
                               qty=str(round(usdt_amount, 2)))
    mark = "OK" if r["retCode"] == 0 else "ERROR"

    if r["retCode"] == 0:
        buy_spent[symbol] = buy_spent.get(symbol, 0) + usdt_amount

    print(f"  [{mark}] {symbol}  {usdt_amount:.2f} USDT  {r.get('retMsg', '')}")


def sell_coin(symbol, qty):
    f   = get_filters(symbol)
    qty = round(math.floor(qty / f["step"]) * f["step"], f["decimals"])
    qty = min(qty, f["max_qty"])

    if qty <= 0:
        print(f"  [skip] {symbol}: no ability to sell - less then minimum order")
        return

    r    = session.place_order(category="spot", symbol=symbol, side="Sell",
                               orderType="Market", qty=str(qty))
    mark = "OK" if r["retCode"] == 0 else "ERROR"
    print(f"  [{mark}] {symbol}  qty={qty}  {r.get('retMsg', '')}")


def get_balance():
    coins  = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["coin"]
    result = {}
    for c in coins:
        qty = float(c["equity"])
        if qty > 0:
            result[c["coin"]] = qty
    return result


# menu

def calculate_portfolio():
    global chosen_portfolio, chosen_budget

    days   = int(input("  Days data [recommended: 30]: ").strip() or 30)
    days   = min(days, 40)
    n      = int(input("  Coins (taken as top by turnover) [recommended: 300]: ").strip() or 50)
    budget = float(input("  The portfolio budget: ").strip() or 10000)
    z = int(input("Kline target, mins [recommended - 60]"))
    print(f"  uploading-{n} coins...")
    pairs = get_top_pairs(n)

    print(f"  getting bars ({days} days, kline = {z} mins)...")
    klines_data = get_klines(pairs, days=days, interval=z)
    df          = build_dataframe(klines_data)

    n_assets  = len(df.columns)
    n_candles = len(df)
    print(f"  Монет: {n_assets},  свечей: {n_candles}")

    if n_candles <= n_assets:
        print(f"  Error: not enough caldles ({n_candles}) for {n_assets} coins — increase days sample")
        return

    portfolios = build_frontier(df)

    for p in portfolios:
        print(f"  [{p['n']:>2}]  ret={p['ret']:+.6f}  std={p['stderr']:.8f}  assets={len(p['weights'])}")

    show_plot(portfolios)

    idx              = int(input("  № of portfolio: ")) - 1
    chosen_portfolio = portfolios[idx]
    chosen_budget    = budget

    print("")
    print(f"  Portfolio #{chosen_portfolio['n']}")
    print(f"  Expected return: {chosen_portfolio['ret']:+.6f}")
    print(f"  Std. err.:   {chosen_portfolio['stderr']:.8f}")

    for sym, w in sorted(chosen_portfolio["weights"].items(), key=lambda x: -abs(x[1])):
        print(f"    {sym:<16}  w={w:+.4f}  {w * budget:+.2f} USDT")

    print("  [saved to cache]")


def buy_portfolio():
    if chosen_portfolio is None:
        print("  At first calculate the portfolio.")
        return

    pos_weights = {}
    for s, w in chosen_portfolio["weights"].items():
        if w > 0:
            pos_weights[s] = w

    total = sum(pos_weights.values())

    ans = input(f"  Buy {len(pos_weights)} coins for {chosen_budget:.2f} USDT? (y/n): ")
    if ans.strip().lower() not in ("да", "y", "yes"):
        return

    buy_spent.clear()

    for symbol, w in pos_weights.items():
        amount = round((w / total) * chosen_budget, 2)
        buy_coin(symbol, amount)


def sell_portfolio():
    if chosen_portfolio is None:
        print("  Calculate the portfolio firstly.")
        return

    ans = input("  Cash out? (y/n): ")
    if ans.strip().lower() not in ("да", "y", "yes"):
        return

    bal         = get_balance()
    usdt_before = bal.get("USDT", 0)
    total_spent = 0.0

    for symbol, w in chosen_portfolio["weights"].items():
        if w <= 0:
            continue
        coin = symbol.replace("USDT", "")
        qty  = bal.get(coin, 0)
        if qty <= 0:
            print(f"  [skip] {symbol}: not on the wallet")
            continue
        sell_coin(symbol, qty)
        total_spent += buy_spent.get(symbol, 0)

    usdt_after = get_balance().get("USDT", 0)
    total_got  = usdt_after - usdt_before
    total_pnl  = total_got - total_spent

    sign = "+" if total_pnl >= 0 else ""
    print(f"  Spent : {total_spent:.2f} USDT")
    print(f"  Gained  : {total_got:.2f} USDT")
    print(f"  Overall PnL       : {sign}{total_pnl:.2f} USDT  ({sign}{total_pnl / total_spent * 100 if total_spent else 0:.2f}%)")

    buy_spent.clear()


def show_balance():
    bal   = get_balance()
    total = 0.0

    print(f"  {'Coin':<12}  {'Amount':>16}  {'USD':>12}")
    print("  " + "-" * 44)

    for coin, qty in bal.items():
        if coin == "USDT":
            usd = qty
        else:
            try:
                price = float(session.get_tickers(category="spot",
                    symbol=coin + "USDT")["result"]["list"][0]["lastPrice"])
                usd = qty * price
            except:
                usd = 0
        total += usd
        print(f"  {coin:<12}  {qty:>16.6f}  ${usd:>11.2f}")

    print("  " + "-" * 44)
    print(f"  {'Generally':<12}  {'':>16}  ${total:>11.2f}")


def sell_all_to_usdt():
    bal   = get_balance()
    coins = []
    for c in bal:
        if c != "USDT":
            coins.append(c)

    if not coins:
        print("  No assets besides USDT.")
        return

    ans = input(f"  Cash out {', '.join(coins)}? (y/n): ")
    if ans.strip().lower() not in ("да", "y", "yes"):
        return

    for coin in coins:
        sell_coin(coin + "USDT", bal[coin])

# running

menu = {
    "1": ("Calculate portfolio", calculate_portfolio),
    "2": ("Buy",              buy_portfolio),
    "3": ("Cash out",             sell_portfolio),
    "4": ("Balance",              show_balance),
    "5": ("Full cash out to USDT",          sell_all_to_usdt),
    "0": ("Exit",               None),
}

print("Bybit Portfolio Manager [DEMO]")

while True:
    print()
    for key, (label, _) in menu.items():
        print(f"  [{key}] {label}")

    choice = input("Choice: ").strip()

    if choice == "0":
        break
    elif choice in menu and menu[choice][1] is not None:
        try:
            menu[choice][1]()
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  Invalid input.")
