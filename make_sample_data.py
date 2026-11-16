"""Create the sample order lines for the market basket recommender.

Writes two files into data/ (fixed random seed, so every run gives the same data):

    order_lines.csv.gz       about 20,000 orders over 2025 for a made-up office and
                             electronics retailer, "Kaveri Office & Electronics", Chennai
    planted_affinities.csv   the buying patterns planted below, so a test can check
                             that the analysis finds them again

Planted on purpose:

  * True affinities of different strengths. A "trigger" product pulls an add-on
    into the same order with a set probability: Laser Printer -> Toner Cartridge
    (55%), Business Laptop -> Laptop Bag + Wireless Mouse together (20%), down to
    weak ones such as Monitor -> Monitor Arm (7%).
  * Popular items that go with everything only by chance. A4 paper, pens,
    notebooks, pen drives and extension boards are dropped into any order with a
    fixed probability, whatever else is in it. They co-occur with every product
    very often, yet no more often than their normal rate. A good analysis must
    not report them as affinities.
  * Seasonality. More orders in March (year-end budgets), June (schools and
    colleges reopen) and October-November (festive season), fewer on Sundays, and
    some categories sell better in some months.
  * A few bad rows: returns with a negative quantity, lines with no product name,
    lines with no price and repeated lines within an order.

Usage:
    python make_sample_data.py

Author : Swaminathan Narayanan
Repo   : https://github.com/Swami529796/market-basket-recommender
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2025
N_ORDERS = 20_000
HERE = Path(__file__).resolve().parent

# name, category, list price in Rs, weight (how often it starts an order)
PRODUCTS = [
    ("Business Laptop", "Computers", 54990, 3.0),
    ("Desktop PC", "Computers", 42990, 1.6),
    ("Tablet 10 inch", "Computers", 18990, 2.0),
    ("Monitor 24 inch", "Computers", 11490, 2.4),
    ("Laptop Bag", "Computer Accessories", 1490, 1.6),
    ("Wireless Mouse", "Computer Accessories", 699, 2.6),
    ("Keyboard", "Computer Accessories", 899, 2.0),
    ("Laptop Cooling Pad", "Computer Accessories", 1190, 0.8),
    ("Laptop Stand", "Computer Accessories", 1290, 1.0),
    ("Mouse Pad", "Computer Accessories", 199, 1.4),
    ("Tablet Cover", "Computer Accessories", 790, 0.8),
    ("Stylus Pen", "Computer Accessories", 1490, 0.6),
    ("USB Hub", "Computer Accessories", 990, 1.4),
    ("Webcam", "Computer Accessories", 2490, 1.5),
    ("Laser Printer", "Printers", 16990, 2.0),
    ("Inkjet Printer", "Printers", 7990, 1.8),
    ("Label Printer", "Printers", 9490, 0.9),
    ("Document Scanner", "Printers", 12990, 0.9),
    ("Toner Cartridge", "Printer Supplies", 3290, 3.2),
    ("Ink Cartridge Set", "Printer Supplies", 1890, 2.8),
    ("Label Rolls", "Printer Supplies", 590, 1.2),
    ("Photo Paper Pack", "Printer Supplies", 390, 0.9),
    ("Wi-Fi Router", "Networking", 2490, 1.8),
    ("Ethernet Cable 5m", "Networking", 249, 1.6),
    ("Network Switch 8 Port", "Networking", 1790, 0.9),
    ("Wi-Fi Range Extender", "Networking", 1690, 1.0),
    ("Headset", "Audio & Video", 1990, 2.2),
    ("Bluetooth Speaker", "Audio & Video", 2290, 1.5),
    ("HDMI Cable", "Audio & Video", 349, 1.8),
    ("Projector", "Audio & Video", 38990, 0.8),
    ("Projector Screen", "Audio & Video", 4990, 0.5),
    ("Conference Speakerphone", "Audio & Video", 7990, 0.7),
    ("External Hard Drive 1TB", "Storage", 4490, 2.0),
    ("Pen Drive 64GB", "Storage", 549, 0.0),
    ("SSD 500GB", "Storage", 3990, 1.4),
    ("Memory Card 128GB", "Storage", 890, 1.3),
    ("Hard Drive Case", "Storage", 349, 0.5),
    ("A4 Copier Paper (Ream)", "Stationery", 290, 0.0),
    ("Ball Pens (Box of 20)", "Stationery", 180, 0.0),
    ("Notebook Pack", "Stationery", 240, 0.0),
    ("Sticky Notes", "Stationery", 120, 1.8),
    ("Stapler", "Stationery", 260, 1.5),
    ("Staples (Box)", "Stationery", 60, 1.4),
    ("Whiteboard Markers", "Stationery", 220, 1.6),
    ("Highlighters", "Stationery", 150, 1.6),
    ("Box Files (Pack of 5)", "Stationery", 450, 1.5),
    ("Envelopes (Pack of 100)", "Stationery", 130, 1.3),
    ("Office Chair", "Office Furniture", 6990, 1.8),
    ("Office Desk", "Office Furniture", 9990, 1.2),
    ("Filing Cabinet", "Office Furniture", 7490, 0.8),
    ("Whiteboard 4x3 ft", "Office Furniture", 2190, 1.2),
    ("Monitor Arm", "Office Furniture", 2890, 0.6),
    ("Paper Shredder", "Office Equipment", 5490, 0.9),
    ("Shredder Oil", "Office Equipment", 390, 0.5),
    ("Laminator", "Office Equipment", 3290, 0.9),
    ("Laminating Pouches", "Office Equipment", 420, 0.9),
    ("Calculator", "Office Equipment", 490, 1.5),
    ("Extension Board", "Office Equipment", 590, 0.0),
    ("UPS 600VA", "Office Equipment", 3490, 1.3),
    ("Whiteboard Duster", "Office Equipment", 90, 0.5),
]

# Popular items: added to ANY order with this probability, whatever else is in it.
POPULAR = {
    "A4 Copier Paper (Ream)": 0.22,
    "Ball Pens (Box of 20)": 0.14,
    "Pen Drive 64GB": 0.09,
    "Notebook Pack": 0.08,
    "Extension Board": 0.07,
}

# trigger, add-on(s) bought with it, probability, strength label
AFFINITIES = [
    ("Laser Printer", ("Toner Cartridge",), 0.55, "strong"),
    ("Inkjet Printer", ("Ink Cartridge Set",), 0.50, "strong"),
    ("Label Printer", ("Label Rolls",), 0.50, "strong"),
    ("Laminator", ("Laminating Pouches",), 0.50, "strong"),
    ("Whiteboard 4x3 ft", ("Whiteboard Markers",), 0.45, "strong"),
    ("Tablet 10 inch", ("Tablet Cover",), 0.40, "strong"),
    ("Paper Shredder", ("Shredder Oil",), 0.40, "strong"),
    ("Business Laptop", ("Laptop Bag", "Wireless Mouse"), 0.20, "strong"),
    ("Business Laptop", ("Laptop Bag",), 0.18, "medium"),
    ("Business Laptop", ("Wireless Mouse",), 0.15, "medium"),
    ("Stapler", ("Staples (Box)",), 0.35, "medium"),
    ("Monitor 24 inch", ("HDMI Cable",), 0.30, "medium"),
    ("Desktop PC", ("Monitor 24 inch",), 0.30, "medium"),
    ("Projector", ("Projector Screen",), 0.30, "medium"),
    ("Wi-Fi Router", ("Ethernet Cable 5m",), 0.28, "medium"),
    ("Desktop PC", ("Keyboard",), 0.25, "medium"),
    ("Projector", ("HDMI Cable",), 0.25, "medium"),
    ("Whiteboard 4x3 ft", ("Whiteboard Duster",), 0.25, "medium"),
    ("Webcam", ("Headset",), 0.22, "medium"),
    ("Office Desk", ("Office Chair",), 0.22, "medium"),
    ("External Hard Drive 1TB", ("Hard Drive Case",), 0.10, "weak"),
    ("Tablet 10 inch", ("Stylus Pen",), 0.10, "weak"),
    ("Inkjet Printer", ("Photo Paper Pack",), 0.10, "weak"),
    ("Business Laptop", ("Laptop Cooling Pad",), 0.08, "weak"),
    ("Monitor 24 inch", ("Monitor Arm",), 0.07, "weak"),
]

MONTH_VOLUME = {1: 0.90, 2: 0.90, 3: 1.25, 4: 1.00, 5: 0.90, 6: 1.20,
                7: 1.00, 8: 0.95, 9: 1.00, 10: 1.35, 11: 1.20, 12: 0.90}
CATEGORY_SEASON = {
    "Stationery": {6: 1.5, 7: 1.2},
    "Computers": {6: 1.2, 10: 1.5, 11: 1.4},
    "Audio & Video": {10: 1.4, 11: 1.3},
    "Office Furniture": {3: 1.5, 4: 1.2},
    "Office Equipment": {3: 1.3},
}
BULK_CATEGORIES = {"Stationery", "Printer Supplies"}   # often bought several at a time


def build_orders(rng: np.random.Generator) -> pd.DataFrame:
    names = [p[0] for p in PRODUCTS]
    category = {p[0]: p[1] for p in PRODUCTS}
    price = {p[0]: p[2] for p in PRODUCTS}
    base = np.array([p[3] for p in PRODUCTS], dtype=float)

    # the chance that each product starts an order, month by month
    month_p = {}
    for month in range(1, 13):
        weights = base * [CATEGORY_SEASON.get(category[n], {}).get(month, 1.0) for n in names]
        month_p[month] = weights / weights.sum()

    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    day_weight = np.array([MONTH_VOLUME[d.month] * (0.4 if d.dayofweek == 6 else 1.0) for d in days])
    order_days = np.sort(rng.choice(len(days), size=N_ORDERS, p=day_weight / day_weight.sum()))

    add_ons: dict[str, list[tuple[tuple[str, ...], float]]] = {}
    for trigger, extras, probability, _ in AFFINITIES:
        add_ons.setdefault(trigger, []).append((extras, probability))

    rows = []
    for number, day_index in enumerate(order_days, start=100001):
        day = days[day_index]
        n_start = min(1 + rng.poisson(0.5), 4)
        basket = list(rng.choice(names, size=n_start, replace=False, p=month_p[day.month]))
        for trigger in list(basket):
            for extras, probability in add_ons.get(trigger, []):
                if rng.random() < probability:
                    basket.extend(extras)
        for item, probability in POPULAR.items():
            if rng.random() < probability:
                basket.append(item)
        for item in dict.fromkeys(basket):   # one line per product, order kept
            bulk = category[item] in BULK_CATEGORIES
            quantity = int(rng.choice([1, 2, 3, 5], p=[0.5, 0.25, 0.15, 0.10])) if bulk else 1
            unit_price = round(price[item] * 0.95) if rng.random() < 0.08 else price[item]
            rows.append((number, day.date().isoformat(), item, category[item], quantity, unit_price))
    return pd.DataFrame(rows, columns=["order_id", "order_date", "product", "category", "quantity", "unit_price"])


def plant_bad_rows(lines: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the kinds of bad rows a real export has. The analysis must report each one."""
    lines = lines.copy()
    lines["quantity"] = lines["quantity"].astype(object)
    lines["unit_price"] = lines["unit_price"].astype(object)
    picks = rng.choice(len(lines), size=38, replace=False)
    returns, blanks, no_price, repeats = picks[:14], picks[14:20], picks[20:30], picks[30:]
    lines.loc[returns, "quantity"] = -1                       # returns
    lines.loc[blanks, "product"] = ""                         # product name missing
    lines.loc[no_price, "unit_price"] = ""                    # price missing
    lines = pd.concat([lines, lines.loc[repeats]])            # the same line twice
    return lines.sort_values("order_id", kind="stable").reset_index(drop=True)


def main() -> None:
    assert len(PRODUCTS) == 60 and len({p[0] for p in PRODUCTS}) == 60
    rng = np.random.default_rng(SEED)
    lines = plant_bad_rows(build_orders(rng), rng)

    data = HERE / "data"
    data.mkdir(exist_ok=True)
    # mtime 0 keeps the compressed file byte-for-byte the same on every run
    lines.to_csv(data / "order_lines.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    planted = pd.DataFrame([(t, " + ".join(e), p, s) for t, e, p, s in AFFINITIES],
                           columns=["trigger", "add_on", "probability", "strength"])
    planted.to_csv(data / "planted_affinities.csv", index=False)

    size_kb = (data / "order_lines.csv.gz").stat().st_size / 1024
    print(f"Orders:            {lines['order_id'].nunique():,}")
    print(f"Order lines:       {len(lines):,}  ({size_kb:,.0f} KB compressed)")
    print(f"Products:          {len(PRODUCTS)} in {len({p[1] for p in PRODUCTS})} categories")
    print(f"Planted affinities: {len(planted)} "
          f"({', '.join(f'{n} {s}' for s, n in planted['strength'].value_counts().items())})")
    print(f"Saved to:          {data.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
