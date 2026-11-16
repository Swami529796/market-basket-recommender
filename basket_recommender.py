"""Market Basket Analysis and Bundle Recommender.

Reads order lines (order id, product, and optionally date, category, quantity and
price), finds which products are bought together more often than chance, and
turns that into three things a merchandiser can act on:

    Rules             "55% of orders with Laser Printer also contain Toner Cartridge;
                      that is 9 times the normal rate"
    Recommendations   a "customers also bought" list for every product
    Bundles           pairs and triples worth selling together, with an ESTIMATE
                      of the extra revenue under a stated uptake assumption
    Evaluation        an honest test on the last part of the period: one item is
                      hidden from each order and the recommender has to find it;
                      compared with simply recommending the best-sellers

The frequent-itemset search (Apriori) is written out in this file, not imported,
so every step can be read and tested. Rules must clear a minimum support,
confidence, lift and number of orders, and are marked significant only if a
chi-square (or Fisher exact) test survives a correction for the number of rules
tested. Lift is what stops a best-seller from appearing to "go with" everything.

Usage:
    python basket_recommender.py
    python basket_recommender.py --level category
    python basket_recommender.py --lookup "Laser Printer"
    python basket_recommender.py --input my_orders.csv --product-col item --order-col invoice_no
    python basket_recommender.py --min-lift 2 --min-orders 30 --uptake 0.05

Settings come from config.yaml; any command-line option overrides the file.

Author : Swaminathan Narayanan
Repo   : https://github.com/Swami529796/market-basket-recommender
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from scipy.stats import chi2_contingency, fisher_exact  # noqa: E402

HERE = Path(__file__).resolve().parent
NAVY, LIGHT_BLUE, AMBER, GREY = "#1F4E78", "#9DB7D5", "#E08A00", "#595959"

DEFAULTS = {
    "input": "data/order_lines.csv.gz",
    "output": "output/basket_analysis.xlsx",
    "charts_dir": "images",
    "columns": {"order": "order_id", "product": "product", "date": "order_date",
                "category": "category", "quantity": "quantity", "price": "unit_price"},
    "level": "product",          # product or category
    "min_support": 0.001,        # share of all orders that contain the items together
    "min_confidence": 0.10,      # share of orders with A that also contain B
    "min_lift": 1.5,             # how many times the normal rate
    "min_orders": 20,            # a rule must rest on at least this many orders
    "min_gain": 0.05,            # a two-item "if" must beat its best one-item rule by this much confidence
    "max_items": 3,              # 2 = pairs only, 3 = pairs and triples
    "alpha": 0.05,               # significance level, after the correction for many tests
    "top_n": 5,                  # recommendations per product
    "holdout": 0.20,             # last share of the orders kept back for the evaluation
    "uptake": 0.10,              # ASSUMPTION: share of near-miss buyers who take the bundle
    "bundle_discount": 0.10,     # ASSUMPTION: discount on the add-on items in a bundle
    "seed": 42,                  # which item is hidden in each held-out order
}

Itemset = tuple[str, ...]


# --------------------------------------------------------------------------- loading

@dataclass
class Orders:
    lines: pd.DataFrame                     # clean lines: order, item, quantity, price
    baskets: dict[str, frozenset[str]]      # order id -> set of items, oldest order first
    issues: pd.DataFrame                    # every row left out or repaired, with the reason
    has_dates: bool = False
    period: str = ""


def load_orders(path: Path, columns: dict[str, str], level: str = "product") -> Orders:
    """Read the order lines and build one basket (set of items) per order.

    Nothing is dropped silently: every row that is left out is counted in `issues`.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    item_col = columns["product"] if level == "product" else columns["category"]
    for needed in (columns["order"], item_col):
        if needed not in raw.columns:
            raise ValueError(f"Column '{needed}' not found in {path.name}. Columns are: {', '.join(raw.columns)}")

    lines = pd.DataFrame({"order": raw[columns["order"]].str.strip(), "item": raw[item_col].str.strip()})
    lines["row"] = np.arange(2, len(raw) + 2)                     # row number as seen in the file
    has = {name: columns.get(name) in raw.columns for name in ("date", "quantity", "price")}
    lines["quantity"] = pd.to_numeric(raw[columns["quantity"]], errors="coerce") if has["quantity"] else 1.0
    lines["price"] = pd.to_numeric(raw[columns["price"]], errors="coerce") if has["price"] else np.nan
    lines["date"] = pd.to_datetime(raw[columns["date"]], errors="coerce") if has["date"] else pd.NaT

    issues: list[dict] = []

    def leave_out(mask: pd.Series, reason: str) -> None:
        nonlocal lines
        if mask.any():
            issues.append({"Problem": reason, "Rows": int(mask.sum()), "Action": "Left out",
                           "Example rows in file": ", ".join(map(str, lines.loc[mask, "row"].head(5)))})
            lines = lines[~mask]

    leave_out(lines["order"] == "", "Order id is blank")
    leave_out(lines["item"] == "", f"{'Product' if level == 'product' else 'Category'} is blank")
    if has["quantity"]:
        leave_out(lines["quantity"].isna(), "Quantity is not a number")
        leave_out(lines["quantity"] <= 0, "Quantity is zero or negative (return or cancellation)")
    if has["price"]:
        bad_price = lines["price"].isna() | (lines["price"] <= 0)
        if bad_price.any():
            issues.append({"Problem": "Price is missing or not positive", "Rows": int(bad_price.sum()),
                           "Action": "Kept in the basket; not used for the typical price",
                           "Example rows in file": ", ".join(map(str, lines.loc[bad_price, "row"].head(5)))})
            lines.loc[bad_price, "price"] = np.nan
    if has["date"] and lines["date"].isna().any():
        bad_date = lines["date"].isna()
        issues.append({"Problem": "Order date is missing or unreadable", "Rows": int(bad_date.sum()),
                       "Action": "Kept; the order is treated as the oldest in the file",
                       "Example rows in file": ", ".join(map(str, lines.loc[bad_date, "row"].head(5)))})
    if level == "product":
        repeated = lines.duplicated(["order", "item"], keep="first")
        if repeated.any():
            issues.append({"Problem": "Same product listed more than once in an order", "Rows": int(repeated.sum()),
                           "Action": "Merged into one line (quantities added)",
                           "Example rows in file": ", ".join(map(str, lines.loc[repeated, "row"].head(5)))})
    lines = (lines.groupby(["order", "item"], sort=False)
             .agg(quantity=("quantity", "sum"), price=("price", "median"), date=("date", "min"), row=("row", "min"))
             .reset_index())
    if lines.empty:
        raise ValueError(f"No usable order lines in {path.name}")

    # oldest order first; with no date column the order of the file is used
    first = lines.groupby("order", sort=False).agg(date=("date", "min"), row=("row", "min"))
    first = first.sort_values(["date", "row"], na_position="first", kind="stable")
    grouped = lines.groupby("order", sort=False)["item"].agg(frozenset)
    baskets = {order: grouped[order] for order in first.index}

    dates = lines["date"].dropna()
    period = f"{dates.min():%d-%b-%Y} to {dates.max():%d-%b-%Y}" if len(dates) else "no order dates"
    issue_table = pd.DataFrame(issues, columns=["Problem", "Rows", "Action", "Example rows in file"])
    return Orders(lines=lines, baskets=baskets, issues=issue_table, has_dates=bool(len(dates)), period=period)


# --------------------------------------------------------------------------- Apriori

def generate_candidates(frequent: set[Itemset], size: int) -> set[Itemset]:
    """Build candidate itemsets of `size` items from the frequent itemsets one item smaller.

    Join: two itemsets that agree on everything but their last item are merged.
    Prune: a candidate is kept only if EVERY smaller subset is itself frequent.
    That is the Apriori idea: a set cannot be common unless all its parts are.
    """
    ordered = sorted(frequent)
    candidates: set[Itemset] = set()
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if left[:-1] != right[:-1]:
                break                      # sorted, so no later itemset shares the prefix
            candidate = left + (right[-1],)
            if all(subset in frequent for subset in combinations(candidate, size - 1)):
                candidates.add(candidate)
    return candidates


def frequent_itemsets(baskets: list[frozenset[str]], min_count: int, max_items: int = 3) -> dict[Itemset, int]:
    """Apriori: every itemset (as a sorted tuple) found in at least `min_count` baskets, with its count."""
    singles = Counter(item for basket in baskets for item in basket)
    frequent = {(item,): count for item, count in singles.items() if count >= min_count}
    found = dict(frequent)
    size = 2
    while frequent and size <= max_items:
        candidates = generate_candidates(set(frequent), size)
        if not candidates:
            break
        useful = {item for candidate in candidates for item in candidate}
        counts: Counter[Itemset] = Counter()
        for basket in baskets:
            items = sorted(useful & basket)
            if len(items) >= size:
                counts.update(combo for combo in combinations(items, size) if combo in candidates)
        frequent = {itemset: count for itemset, count in counts.items() if count >= min_count}
        found.update(frequent)
        size += 1
    return found


# --------------------------------------------------------------------------- rules

def rule_metrics(n: int, n_a: int, n_c: int, n_both: int) -> dict[str, float]:
    """Measures for the rule "orders with A also contain C".

    n       all orders            n_a     orders with A
    n_c     orders with C         n_both  orders with A and C
    """
    support = n_both / n                      # share of all orders with both
    confidence = n_both / n_a                 # share of A's orders that also have C
    normal_rate = n_c / n                     # share of all orders with C
    lift = confidence / normal_rate           # 1 = no link; 3 = three times the normal rate
    leverage = support - (n_a / n) * normal_rate          # extra share of orders above chance
    conviction = math.inf if confidence >= 1 else (1 - normal_rate) / (1 - confidence)
    return {"support": support, "confidence": confidence, "normal_rate": normal_rate,
            "lift": lift, "leverage": leverage, "conviction": conviction}


def significance(n: int, n_a: int, n_c: int, n_both: int) -> tuple[float, str]:
    """P-value that A and C are unrelated, from the 2x2 table of orders with / without each.

    Chi-square normally; Fisher's exact test when an expected count is under 5,
    where the chi-square approximation is not reliable.
    """
    table = np.array([[n_both, n_a - n_both], [n_c - n_both, n - n_a - n_c + n_both]])
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    if (expected == 0).any():
        return 1.0, "none"                    # A or C is in every order (or none): nothing to test
    if expected.min() < 5:
        return float(fisher_exact(table)[1]), "Fisher exact"
    return float(chi2_contingency(table)[1]), "chi-square"


def sentence(antecedent: Itemset, consequent: str, confidence: float, lift: float, normal_rate: float) -> str:
    return (f"{confidence:.0%} of orders with {' and '.join(antecedent)} also contain {consequent}; "
            f"that is {lift:.1f} times the normal rate ({normal_rate:.1%}).")


RULE_COLUMNS = ["antecedent", "consequent", "orders_together", "support", "confidence", "normal_rate", "lift",
                "leverage", "conviction", "p_value", "test", "significant", "confidence_gain", "gain_p_value", "adds_nothing", "passes", "sentence"]


def build_rules(itemsets: dict[Itemset, int], n: int, min_confidence: float, min_lift: float,
                alpha: float = 0.05, min_gain: float = 0.05) -> pd.DataFrame:
    """Every rule "antecedent -> one more item" from the frequent itemsets, with its measures.

    `significant` uses a Bonferroni correction: the p-value must be below alpha
    divided by the number of rules tested, because testing hundreds of rules at 5%
    each would let several through by luck alone.

    `adds_nothing` applies to rules with two (or more) items in the "if" part.
    "Paper + Shredder -> Shredder Oil" is just "Shredder -> Shredder Oil" with a
    best-seller riding along, and lift alone does not catch that. So the rule is
    compared with each shorter rule inside it (drop one item from the "if"):
      * confidence_gain  its confidence minus the best shorter rule's; must be at
                         least `min_gain`
      * gain_p_value     among orders that already have the shorter "if", does
                         the extra item really change how often the consequent is
                         bought? Same test and correction as above.

    `passes` = confidence and lift thresholds met, and not `adds_nothing`.
    """
    rows = []
    for itemset, n_both in itemsets.items():
        if len(itemset) < 2:
            continue
        for consequent in itemset:
            antecedent = tuple(item for item in itemset if item != consequent)
            n_a, n_c = itemsets[antecedent], itemsets[(consequent,)]
            p_value, test = significance(n, n_a, n_c, n_both)
            gain, gain_p = np.nan, np.nan
            if len(antecedent) > 1:
                best_shorter, gain_p = 0.0, 0.0
                for dropped in antecedent:
                    shorter = tuple(item for item in antecedent if item != dropped)
                    n_shorter = itemsets[shorter]
                    n_shorter_c = itemsets[tuple(sorted(shorter + (consequent,)))]
                    best_shorter = max(best_shorter, n_shorter_c / n_shorter)
                    gain_p = max(gain_p, significance(n_shorter, n_a, n_shorter_c, n_both)[0])
                gain = n_both / n_a - best_shorter
            rows.append({"antecedent": antecedent, "consequent": consequent, "orders_together": n_both,
                         **rule_metrics(n, n_a, n_c, n_both), "p_value": p_value, "test": test,
                         "confidence_gain": gain, "gain_p_value": gain_p})
    rules = pd.DataFrame(rows, columns=RULE_COLUMNS)
    if rules.empty:
        return rules
    threshold = alpha / len(rules)
    rules["significant"] = rules["p_value"] < threshold
    longer = rules["antecedent"].map(len) > 1
    rules["adds_nothing"] = longer & ((rules["confidence_gain"] < min_gain) | (rules["gain_p_value"] >= threshold))
    rules["passes"] = ((rules["confidence"] >= min_confidence) & (rules["lift"] >= min_lift)
                       & ~rules["adds_nothing"])
    rules["sentence"] = [sentence(a, c, conf, lift, rate) for a, c, conf, lift, rate in
                         zip(rules["antecedent"], rules["consequent"], rules["confidence"], rules["lift"],
                             rules["normal_rate"])]
    return rules.sort_values(["lift", "orders_together"], ascending=False, kind="stable").reset_index(drop=True)


def mine(baskets: list[frozenset[str]], cfg: dict) -> tuple[dict[Itemset, int], pd.DataFrame]:
    """Frequent itemsets and all their rules for one set of baskets."""
    n = len(baskets)
    min_count = max(int(cfg["min_orders"]), math.ceil(cfg["min_support"] * n), 1)
    itemsets = frequent_itemsets(baskets, min_count, int(cfg["max_items"]))
    return itemsets, build_rules(itemsets, n, cfg["min_confidence"], cfg["min_lift"], cfg["alpha"], cfg["min_gain"])


def reliable(rules: pd.DataFrame) -> pd.DataFrame:
    """The rules used for recommendations and bundles: past the thresholds AND significant."""
    return rules[rules["passes"] & rules["significant"]]


# --------------------------------------------------------------------------- recommender

@dataclass
class Recommender:
    """Ranks products to suggest for a basket.

    methods
      best_sellers   the most-ordered products, whatever is in the basket (the baseline)
      rules          products pointed to by a reliable rule, highest confidence first
      blended        every product gets a score: the rule confidence where a rule
                     applies, otherwise its normal rate; highest score first
    """
    rules: pd.DataFrame
    popularity: dict[str, float]                      # item -> share of orders containing it
    by_antecedent: dict[frozenset, list[tuple[str, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for antecedent, consequent, confidence in zip(self.rules["antecedent"], self.rules["consequent"],
                                                      self.rules["confidence"]):
            self.by_antecedent.setdefault(frozenset(antecedent), []).append((consequent, confidence))
        self.max_size = max((len(a) for a in self.by_antecedent), default=0)
        self.ranked_sellers = sorted(self.popularity, key=lambda item: (-self.popularity[item], item))

    def rule_scores(self, basket: frozenset[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        items = sorted(basket)
        for size in range(1, min(self.max_size, len(items)) + 1):
            for combo in combinations(items, size):
                for consequent, confidence in self.by_antecedent.get(frozenset(combo), ()):
                    if consequent not in basket and confidence > scores.get(consequent, 0.0):
                        scores[consequent] = confidence
        return scores

    def recommend(self, basket: frozenset[str], n: int, method: str = "blended") -> list[str]:
        if method == "best_sellers":
            return [item for item in self.ranked_sellers if item not in basket][:n]
        scores = self.rule_scores(basket)
        if method == "blended":
            scores = {item: max(rate, scores.get(item, 0.0)) for item, rate in self.popularity.items()
                      if item not in basket}
        elif method != "rules":
            raise ValueError(f"Unknown method: {method}")
        return sorted(scores, key=lambda item: (-scores[item], item))[:n]


def popularity_of(baskets: list[frozenset[str]]) -> dict[str, float]:
    counts = Counter(item for basket in baskets for item in basket)
    return {item: count / len(baskets) for item, count in counts.items()}


def recommendation_table(rules: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """"Customers also bought": the top N single-product rules for each product."""
    single = rules[rules["antecedent"].map(len) == 1].copy()
    single["product"] = single["antecedent"].str[0]
    single = single.sort_values(["product", "confidence", "lift"], ascending=[True, False, False], kind="stable")
    single["rank"] = single.groupby("product").cumcount() + 1
    single = single[single["rank"] <= top_n]
    return single[["product", "rank", "consequent", "confidence", "lift", "orders_together", "sentence"]] \
        .rename(columns={"consequent": "also_bought"}).reset_index(drop=True)


# --------------------------------------------------------------------------- bundles

def suggest_bundles(itemsets: dict[Itemset, int], rules: pd.DataFrame, orders: Orders, min_lift: float,
                    uptake: float, discount: float) -> pd.DataFrame:
    """Pairs and triples worth selling as a bundle, with an ESTIMATE of extra revenue.

    A set qualifies only if every item in it is pulled by the others: each of its
    rules has a lift of at least `min_lift`, is significant and, for a triple, adds
    something over the pairs inside it. So a best-seller cannot ride along. The estimate, for the period covered by the data:

      anchor            the most expensive item; the bundle is offered to its buyers
      near-miss orders  orders with the anchor but not all of the add-on items
      extra revenue     uptake x value of the missing add-ons, sold at the discount
      discount given    the discount handed to buyers who already take the full set
      net estimate      extra revenue - discount given

    A pair is also left out when it is linked only through a third product: laptop
    bags and cooling pads are bought together because both go with a laptop, and
    among laptop orders they are unrelated. The test is the pair's lift counted
    inside the orders that contain the third product; under `min_lift` means the
    third product explains the link. A triple containing such a pair is left out
    too, unless the third product is part of the triple.
    """
    columns = ["bundle", "type", "anchor", "bundle_price", "anchor_orders", "already_complete", "attach_rate",
               "near_miss_orders", "lowest_lift", "extra_revenue", "discount_given", "net_estimate"]
    price = orders.lines.groupby("item")["price"].median().dropna().to_dict()
    lifts: dict[Itemset, list[float]] = {}      # itemset -> lift of each of its rules (-1 = rule not good enough)
    good_enough = rules["significant"] & (rules["lift"] >= min_lift) & ~rules["adds_nothing"]
    for antecedent, consequent, lift, good in zip(rules["antecedent"], rules["consequent"], rules["lift"],
                                                  good_enough):
        key = tuple(sorted(antecedent + (consequent,)))
        lifts.setdefault(key, []).append(lift if good else -1.0)

    explained_by: dict[Itemset, set[str]] = {}   # pair -> third products that explain it
    for triple, n_all in itemsets.items():
        if len(triple) != 3:
            continue
        for third in triple:
            first, second = (item for item in triple if item != third)
            n_first, n_second = itemsets[tuple(sorted((first, third)))], itemsets[tuple(sorted((second, third)))]
            if n_all * itemsets[(third,)] / (n_first * n_second) < min_lift:
                explained_by.setdefault((first, second), set()).add(third)

    containing: dict[str, list[frozenset[str]]] = {}
    rows = []
    for itemset, lift_list in lifts.items():
        if min(lift_list) < 0 or any(item not in price for item in itemset):
            continue
        if any(explained_by.get(pair, set()) - set(itemset) for pair in combinations(itemset, 2)):
            continue
        anchor = max(itemset, key=lambda item: (price[item], item))
        add_ons = [item for item in itemset if item != anchor]
        if anchor not in containing:
            containing[anchor] = [basket for basket in orders.baskets.values() if anchor in basket]
        missing_value, near_miss = 0.0, 0
        for basket in containing[anchor]:
            missing = [item for item in add_ons if item not in basket]
            if missing:
                near_miss += 1
                missing_value += sum(price[item] for item in missing)
        complete = itemsets[itemset]
        extra = uptake * missing_value * (1 - discount)
        given = discount * sum(price[item] for item in add_ons) * complete
        rows.append({"bundle": " + ".join([anchor] + sorted(add_ons, key=lambda item: -price[item])),
                     "type": "Pair" if len(itemset) == 2 else "Triple", "anchor": anchor,
                     "bundle_price": sum(price[item] for item in itemset),
                     "anchor_orders": len(containing[anchor]), "already_complete": complete,
                     "attach_rate": complete / len(containing[anchor]), "near_miss_orders": near_miss,
                     "lowest_lift": min(lift_list), "extra_revenue": extra, "discount_given": given,
                     "net_estimate": extra - given})
    bundles = pd.DataFrame(rows, columns=columns)
    return bundles.sort_values("net_estimate", ascending=False, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- evaluation

METHODS = {"best_sellers": "Best-sellers (baseline)", "rules": "Rules only", "blended": "Rules + best-sellers"}


def evaluate(baskets: list[frozenset[str]], cfg: dict, cutoffs: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Hold out the last part of the period and test the recommender on it.

    Rules are learned from the earlier orders only. From every held-out order with
    two or more items, one item is hidden (chosen at random, fixed seed) and each
    method is asked for its top N given the rest. A "hit" is the hidden item
    appearing in that list.
    """
    split = len(baskets) - int(round(len(baskets) * cfg["holdout"]))
    train, test = baskets[:split], baskets[split:]
    _, train_rules = mine(train, cfg)
    recommender = Recommender(reliable(train_rules), popularity_of(train))
    rng = np.random.default_rng(int(cfg["seed"]))

    top = max(cutoffs)
    hits = {method: Counter() for method in METHODS}
    cases = covered = 0
    covered_hits = {method: 0 for method in METHODS}
    headline = int(cfg["top_n"])
    for basket in test:
        if len(basket) < 2:
            continue
        items = sorted(basket)
        hidden = items[int(rng.integers(len(items)))]
        rest = basket - {hidden}
        cases += 1
        has_rule = bool(recommender.rule_scores(rest))
        covered += has_rule
        for method in METHODS:
            ranked = recommender.recommend(rest, max(top, headline), method)
            position = ranked.index(hidden) + 1 if hidden in ranked else None
            for cutoff in cutoffs:
                hits[method][cutoff] += position is not None and position <= cutoff
            if has_rule and position is not None and position <= headline:
                covered_hits[method] += 1

    table = pd.DataFrame([{"method": label, **{f"top_{c}": hits[m][c] / cases if cases else np.nan for c in cutoffs}}
                          for m, label in METHODS.items()])
    by_coverage = pd.DataFrame([{"method": label,
                                 "hit_rate_rule_applies": covered_hits[m] / covered if covered else np.nan,
                                 "hit_rate_no_rule": ((hits[m][headline] - covered_hits[m]) / (cases - covered)
                                                      if cases > covered and headline in cutoffs else np.nan)}
                                for m, label in METHODS.items()])
    return {"table": table, "by_coverage": by_coverage, "train_orders": len(train), "test_orders": len(test),
            "cases": cases, "covered": covered, "train_rules": len(recommender.rules), "cutoffs": cutoffs}


# --------------------------------------------------------------------------- charts

def tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    ax.tick_params(colors=GREY)


def chart_rules(rules: pd.DataFrame, popularity: dict[str, float], cfg: dict, path: Path, noun: str) -> None:
    """Lift against support for every pair of items; the strongest pairs are labelled."""
    pairs = rules[rules["antecedent"].map(len) == 1].copy()
    if pairs.empty:
        return
    pairs["pair"] = [tuple(sorted((a[0], c))) for a, c in zip(pairs["antecedent"], pairs["consequent"])]
    pairs["good"] = pairs["passes"] & pairs["significant"]
    pairs = pairs.groupby("pair").agg(support=("support", "first"), lift=("lift", "first"), good=("good", "any"),
                                      confidence=("confidence", "max")).reset_index()
    sellers = set(sorted(popularity, key=popularity.get, reverse=True)[:5])
    pairs["seller"] = pairs["pair"].map(lambda pair: bool(sellers & set(pair))) & ~pairs["good"]
    other = pairs[~pairs["good"] & ~pairs["seller"]]
    seller, good = pairs[pairs["seller"]], pairs[pairs["good"]]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(other["support"] * 100, other["lift"], s=26, color=LIGHT_BLUE, alpha=0.8, label="Other pairs")
    ax.scatter(seller["support"] * 100, seller["lift"], s=30, color=AMBER, alpha=0.85,
               label=f"Pairs with a top-5 best-selling {noun}")
    ax.scatter(good["support"] * 100, good["lift"], s=44, color=NAVY, label="Reliable affinity (passes every check)")
    ax.axhline(1, color=GREY, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    low, high = ax.get_xlim()
    ax.text(high, 0.9, "lift 1 = bought together no more often than chance ", color=GREY, fontsize=9, ha="right",
            va="top")
    labelled = good.sort_values("lift", ascending=False).head(6)
    middle = labelled["support"].median()
    for _, row in labelled.iterrows():                      # labels go outwards so they do not collide
        left = row["support"] < middle
        ax.annotate(" + ".join(row["pair"]), (row["support"] * 100, row["lift"]), xytext=(-8 if left else 8, -3),
                    textcoords="offset points", fontsize=8.5, color=NAVY, ha="right" if left else "left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("Support: share of all orders that contain both", color=GREY)
    ax.set_ylabel("Lift: times the normal rate", color=GREY)
    near_one = seller["lift"].between(0.67, 1.5).mean() if len(seller) else 0
    title = f"{len(good)} of {len(pairs)} frequent {noun} pairs are real affinities"
    if near_one >= 0.9:
        title += "; best-sellers pair with everything but stay near lift 1"
    ax.set_title(title, color=NAVY, fontsize=12.5, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.82), fontsize=9, labelcolor=GREY)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_evaluation(result: dict, path: Path, noun: str) -> None:
    table, cutoffs = result["table"], result["cutoffs"]
    if not result["cases"]:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colours = [LIGHT_BLUE, AMBER, NAVY]
    width = 0.26
    for i, (_, row) in enumerate(table.iterrows()):
        values = [row[f"top_{c}"] * 100 for c in cutoffs]
        bars = ax.bar(np.arange(len(cutoffs)) + (i - 1) * width, values, width, color=colours[i], label=row["method"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.0f}%", ha="center", fontsize=9,
                    color=GREY)
    ax.set_xticks(np.arange(len(cutoffs)))
    ax.set_xticklabels([f"Top {c}" for c in cutoffs])
    ax.set_ylabel("Hit rate: hidden item was in the list", color=GREY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    base, rules_only, blended = (table[f"top_{cutoffs[-2]}"].iloc[i] * 100 for i in range(3))
    ax.set_title(f"Hidden {noun} found in the top {cutoffs[-2]}: best-sellers {base:.0f}%, rules only "
                 f"{rules_only:.0f}%, both together {blended:.0f}%", color=NAVY, fontsize=12.5, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelcolor=GREY)
    ax.set_ylim(0, max(table[[f"top_{c}" for c in cutoffs]].max()) * 100 * 1.18)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_bundles(bundles: pd.DataFrame, cfg: dict, path: Path) -> None:
    if bundles.empty:
        return
    top = bundles.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.barh(top["bundle"], top["net_estimate"] / 1000, color=[NAVY if t == "Triple" else LIGHT_BLUE
                                                                     for t in top["type"]])
    for bar, value in zip(bars, top["net_estimate"] / 1000):
        ax.text(max(value, 0) + top["net_estimate"].max() / 1000 * 0.01, bar.get_y() + bar.get_height() / 2,
                f"Rs {value:,.0f}k", va="center", fontsize=9, color=GREY)
    ax.set_xlabel("ESTIMATED net extra revenue over the period, Rs thousand\n"
                  f"(assumes {cfg['uptake']:.0%} uptake, {cfg['bundle_discount']:.0%} discount "
                  f"on add-ons; an estimate, not a forecast)", color=GREY)
    fig.suptitle(f"Largest estimated gain: {bundles.iloc[0]['bundle']}", color=NAVY, fontsize=12.5, x=0.02,
                 ha="left")
    ax.set_xlim(min(0, top["net_estimate"].min() / 1000 * 1.1), top["net_estimate"].max() / 1000 * 1.14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=NAVY), plt.Rectangle((0, 0), 1, 1, color=LIGHT_BLUE)]
    ax.legend(handles, ["Triple", "Pair"], frameon=False, loc="lower right", fontsize=9.5, labelcolor=GREY)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- workbook

FORMATS = {"Support": "0.00%", "Confidence": "0.0%", "Normal Rate": "0.0%", "Lift": "0.0", "Leverage": "0.0000",
           "Conviction": "0.00", "P Value": "0.0E+00", "Confidence Gain": "+0.0%;-0.0%", "Attach Rate": "0.0%",
           "Lowest Lift": "0.0", "Bundle Price": "#,##0", "Extra Revenue": "#,##0", "Discount Given": "#,##0",
           "Net Estimate": "#,##0", "Orders Together": "#,##0", "Anchor Orders": "#,##0",
           "Already Complete": "#,##0", "Near Miss Orders": "#,##0"}


def style_sheet(sheet, header_row: int = 1) -> None:
    for cell in sheet[header_row]:
        if cell.value is not None:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", start_color="1F4E78")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.freeze_panes = sheet.cell(row=header_row + 1, column=1)
    headers = {cell.column: cell.value for cell in sheet[header_row]}
    for column in sheet.columns:
        number_format = FORMATS.get(headers.get(column[0].column))
        longest = 0
        for cell in column:
            if cell.row > header_row and number_format and isinstance(cell.value, (int, float)):
                cell.number_format = number_format
            elif cell.row > header_row and isinstance(cell.value, float) and 0 <= cell.value <= 1:
                cell.number_format = "0.0%"
            if cell.row >= header_row and cell.value is not None:
                longest = max(longest, len(str(cell.value)) if isinstance(cell.value, str) else 10)
        sheet.column_dimensions[column[0].column_letter].width = min(max(longest + 2, 10), 95)


def pretty(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns=lambda name: name.replace("_", " ").title())


def write_workbook(path: Path, summary: list[tuple[str, object]], rules: pd.DataFrame, recommendations: pd.DataFrame,
                   bundles: pd.DataFrame, evaluation: dict, issues: pd.DataFrame, cfg: dict, noun: str) -> None:
    shown = rules[rules["passes"]].copy()
    shown["antecedent"] = shown["antecedent"].map(" + ".join)
    shown["significant"] = shown["significant"].map({True: "Yes", False: "No"})
    shown["conviction"] = shown["conviction"].replace(math.inf, np.nan)     # blank = never bought without it
    shown = shown[["sentence", "antecedent", "consequent", "orders_together", "support", "confidence", "normal_rate",
                   "lift", "leverage", "conviction", "p_value", "test", "significant", "confidence_gain"]]
    shown = shown.rename(columns={"sentence": "rule_in_plain_words", "antecedent": "if_the_order_has",
                                  "consequent": "it_also_has"})
    recs = recommendations.rename(columns={"product": noun, "sentence": "in_plain_words"})

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary, columns=["Item", "Value"]).to_excel(writer, sheet_name="Summary", index=False)
        pretty(shown).to_excel(writer, sheet_name="Rules", index=False)
        pretty(recs).to_excel(writer, sheet_name="Recommendations", index=False)

        note = (f"ESTIMATE ONLY, not a forecast. Assumes {cfg['uptake']:.0%} of buyers of the anchor item who did not "
                f"take every add-on would take the bundle, at {cfg['bundle_discount']:.0%} off the add-ons. "
                f"Change with --uptake and --bundle-discount. Amounts in Rs, for the period in the data.")
        if bundles.empty:
            note = ("No bundles: " + ("bundles need product prices, so they are produced at product level only."
                                      if cfg["level"] == "category" or not len(rules)
                                      else "no pair or triple passed every check, or prices are missing."))
        pretty(bundles).to_excel(writer, sheet_name="Bundles", index=False, startrow=2)
        writer.sheets["Bundles"]["A1"] = note
        writer.sheets["Bundles"]["A1"].font = Font(bold=True, color="C00000")

        table = evaluation["table"].rename(columns=lambda c: c.replace("top_", "Hit rate, top ") if "top_" in c else c)
        facts = pd.DataFrame([
            ("Orders used to learn the rules (earliest)", evaluation["train_orders"]),
            ("Orders held out (latest)", evaluation["test_orders"]),
            ("Test cases (held-out orders with 2 or more items; one item hidden)", evaluation["cases"]),
            ("Test cases where at least one rule applied", evaluation["covered"]),
            ("Reliable rules learned from the earlier orders", evaluation["train_rules"]),
        ], columns=["Item", "Value"])
        facts.to_excel(writer, sheet_name="Evaluation", index=False)
        pretty(table).to_excel(writer, sheet_name="Evaluation", index=False, startrow=len(facts) + 3)
        split = evaluation["by_coverage"].rename(columns={
            "hit_rate_rule_applies": f"Top {cfg['top_n']} hit rate when a rule applied",
            "hit_rate_no_rule": f"Top {cfg['top_n']} hit rate when no rule applied"})
        split_row = len(facts) + 3 + len(table) + 3
        pretty(split).to_excel(writer, sheet_name="Evaluation", index=False, startrow=split_row)

        (issues if len(issues) else pd.DataFrame([{"Problem": "None found", "Rows": 0, "Action": "",
                                                    "Example rows in file": ""}])) \
            .to_excel(writer, sheet_name="Data Issues", index=False)

        for name, sheet in writer.sheets.items():
            style_sheet(sheet, header_row=3 if name == "Bundles" else 1)
        sheet = writer.sheets["Evaluation"]
        for header_row in (len(facts) + 4, split_row + 1):
            for cell in sheet[header_row]:
                if cell.value is not None:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", start_color="1F4E78")
        writer.sheets["Rules"].column_dimensions["A"].width = 95
        writer.sheets["Summary"].column_dimensions["A"].width = 58
        writer.sheets["Summary"].column_dimensions["B"].width = 34
        for row in writer.sheets["Summary"].iter_rows(min_row=2, min_col=2, max_col=2):
            row[0].alignment = Alignment(horizontal="left")
            row[0].number_format = "General"


# --------------------------------------------------------------------------- command line

def load_config(args: argparse.Namespace) -> dict:
    cfg = {**DEFAULTS, "columns": dict(DEFAULTS["columns"])}
    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown setting(s) in {config_path.name}: {', '.join(sorted(unknown))}")
        cfg["columns"].update(loaded.pop("columns", None) or {})
        cfg.update(loaded)
        base = config_path.parent
    elif args.config:
        raise ValueError(f"Config file not found: {config_path}")
    else:
        base = HERE
    for key in ("input", "output", "charts_dir"):           # paths in the config are relative to the config file
        cfg[key] = str((base / cfg[key]).resolve()) if not Path(cfg[key]).is_absolute() else cfg[key]
    explicit_output, explicit_charts = args.output is not None, args.charts_dir is not None
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if key != "columns" and value is not None:
            cfg[key] = value
    for name in cfg["columns"]:
        value = getattr(args, f"{name}_col", None)
        if value is not None:
            cfg["columns"][name] = value

    if cfg["level"] not in ("product", "category"):
        raise ValueError("level must be 'product' or 'category'")
    for key in ("min_support", "min_confidence", "min_gain", "holdout", "uptake", "bundle_discount", "alpha"):
        if not 0 <= float(cfg[key]) < 1:
            raise ValueError(f"{key} must be between 0 and 1 (for example 0.1 for 10%), got {cfg[key]}")
    if int(cfg["max_items"]) not in (2, 3, 4):
        raise ValueError("max_items must be 2, 3 or 4")
    if cfg["level"] == "category" and not explicit_output:   # keep the product-level workbook
        out = Path(cfg["output"])
        cfg["output"] = str(out.with_name(f"{out.stem}_category{out.suffix}"))
    if cfg["level"] == "category" and not explicit_charts:   # images/ is kept for the product-level charts
        cfg["charts_dir"] = str(Path(cfg["output"]).parent)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find products bought together and turn them into recommendations.")
    parser.add_argument("--config", help="YAML settings file (default: config.yaml next to this script)")
    parser.add_argument("--input", help="order lines, CSV or CSV.GZ")
    parser.add_argument("--output", help="Excel workbook to write")
    parser.add_argument("--charts-dir", dest="charts_dir", help="folder for the PNG charts")
    parser.add_argument("--no-charts", action="store_true", help="do not write the PNG charts")
    parser.add_argument("--level", choices=["product", "category"], help="analyse products or whole categories")
    for name in DEFAULTS["columns"]:
        parser.add_argument(f"--{name}-col", dest=f"{name}_col", help=f"name of the {name} column")
    parser.add_argument("--min-support", dest="min_support", type=float)
    parser.add_argument("--min-confidence", dest="min_confidence", type=float)
    parser.add_argument("--min-lift", dest="min_lift", type=float)
    parser.add_argument("--min-orders", dest="min_orders", type=int)
    parser.add_argument("--min-gain", dest="min_gain", type=float,
                        help="confidence a two-item rule must add over its best one-item rule")
    parser.add_argument("--max-items", dest="max_items", type=int, help="2 = pairs only, 3 = pairs and triples")
    parser.add_argument("--top-n", dest="top_n", type=int, help="recommendations per product")
    parser.add_argument("--holdout", type=float, help="share of the latest orders held out for the evaluation")
    parser.add_argument("--uptake", type=float, help="assumed share of near-miss buyers who take a bundle")
    parser.add_argument("--bundle-discount", dest="bundle_discount", type=float,
                        help="assumed discount on the add-on items")
    parser.add_argument("--seed", type=int, help="random seed for the hidden item in the evaluation")
    parser.add_argument("--lookup", help='print "customers also bought" for one product and stop')
    return parser


def run(cfg: dict, charts: bool = True, lookup: str | None = None) -> dict:
    """The whole analysis. Returns the tables so tests (or a notebook) can use them."""
    started = time.perf_counter()
    noun = cfg["level"]
    plural = {"product": "products", "category": "categories"}[noun]
    orders = load_orders(Path(cfg["input"]), cfg["columns"], cfg["level"])
    baskets = list(orders.baskets.values())
    n = len(baskets)

    itemsets, rules = mine(baskets, cfg)
    good = reliable(rules)
    recommendations = recommendation_table(good, int(cfg["top_n"]))

    if lookup is not None:
        names = {item.lower(): item for item in popularity_of(baskets)}
        item = names.get(lookup.strip().lower())
        if item is None:
            print(f"'{lookup}' is not a {noun} in {Path(cfg['input']).name}.", file=sys.stderr)
            return {"status": 1}
        found = recommendations[recommendations["product"] == item]
        print(f"Customers who bought {item} also bought:")
        if found.empty:
            print("  (no reliable rule: nothing is bought with it clearly more often than chance)")
        for _, row in found.iterrows():
            print(f"  {row['rank']}. {row['also_bought']:<26} {row['confidence']:.0%} of its orders, "
                  f"{row['lift']:.1f}x the normal rate, {row['orders_together']} orders")
        return {"status": 0, "lookup": found}

    # bundles need a price per item, so they are built at product level only
    bundle_rules = rules if cfg["level"] == "product" else rules.iloc[0:0]
    bundles = suggest_bundles(itemsets, bundle_rules, orders, cfg["min_lift"], cfg["uptake"], cfg["bundle_discount"])
    evaluation = evaluate(baskets, cfg)

    passing = rules[rules["passes"]]
    n_items = len(popularity_of(baskets))
    summary = [
        ("Input file", Path(cfg["input"]).name),
        ("Level analysed", noun),
        ("Period", orders.period),
        ("Orders", n),
        ("Orders with 2 or more items", sum(len(b) >= 2 for b in baskets)),
        (f"Distinct {plural}", n_items),
        ("Rows left out or repaired (see Data Issues)", int(orders.issues["Rows"].sum())),
        ("Minimum support", f"{cfg['min_support']:.2%}"),
        ("Minimum confidence", f"{cfg['min_confidence']:.0%}"),
        ("Minimum lift", cfg["min_lift"]),
        ("Minimum orders behind a rule", cfg["min_orders"]),
        ("Minimum confidence gain for a two-item 'if'", f"{cfg['min_gain']:.0%}"),
        ("Rules tested", len(rules)),
        ("Rules past the thresholds (Rules sheet)", len(passing)),
        ("  of which statistically significant", int(passing["significant"].sum())),
        (f"{plural.title()} with at least one recommendation", recommendations["product"].nunique()),
        ("Bundle candidates", len(bundles)),
        ("Significance rule", f"p-value below {cfg['alpha']} / number of rules tested (Bonferroni)"),
    ]
    write_workbook(Path(cfg["output"]), summary, rules, recommendations, bundles, evaluation, orders.issues, cfg, noun)

    if charts:
        folder = Path(cfg["charts_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "" if cfg["level"] == "product" else "_category"
        chart_rules(rules, popularity_of(baskets), cfg, folder / f"rules_lift_vs_support{suffix}.png", noun)
        chart_evaluation(evaluation, folder / f"hit_rate_vs_baseline{suffix}.png", noun)
        chart_bundles(bundles, cfg, folder / "bundle_estimates.png")

    # ---- console report
    print(f"Input:            {Path(cfg['input']).name}  ({noun} level)")
    print(f"Orders:           {n:,} ({orders.period}), {sum(len(b) >= 2 for b in baskets):,} with 2 or more items, "
          f"{n_items} {plural}")
    for _, issue in orders.issues.iterrows():
        print(f"  ! {issue['Rows']} rows: {issue['Problem']} -> {issue['Action']}")
    print(f"Rules tested:     {len(rules):,}; past the thresholds: {len(passing)}; "
          f"also significant: {int(passing['significant'].sum())}")
    riders = rules[rules["adds_nothing"] & (rules["confidence"] >= cfg["min_confidence"])
                   & (rules["lift"] >= cfg["min_lift"])]
    print(f"                  ({len(riders)} more two-item rules set aside: no better than a one-item rule inside them)")
    print()
    print("Strongest rules:")
    for text in good.sort_values("lift", ascending=False)["sentence"].head(8):
        print(f"  {text}")
    print()
    print(f"Recommendations:  {recommendations['product'].nunique()} of {n_items} {plural} have at least one")
    if len(bundles):
        print(f"Bundles:          {len(bundles)} candidates. ESTIMATE at {cfg['uptake']:.0%} uptake, "
              f"{cfg['bundle_discount']:.0%} discount on add-ons:")
        for _, row in bundles.head(5).iterrows():
            print(f"  {row['bundle']:<52} net Rs {row['net_estimate']:>10,.0f}")
    else:
        print("Bundles:          none (product level with prices is needed)")
    print()
    print(f"Evaluation:       rules learned from the first {evaluation['train_orders']:,} orders, tested on "
          f"{evaluation['cases']:,} held-out orders with one item hidden")
    header = "".join(f"{'top ' + str(c):>9}" for c in evaluation["cutoffs"])
    print(f"  {'':<26}{header}")
    for _, row in evaluation["table"].iterrows():
        print(f"  {row['method']:<26}" + "".join(f"{row[f'top_{c}']:>9.1%}" for c in evaluation["cutoffs"]))
    if evaluation["cases"]:
        print(f"  A rule applied in {evaluation['covered']:,} of {evaluation['cases']:,} test cases "
              f"({evaluation['covered'] / evaluation['cases']:.0%})")
    print()
    print(f"Time taken:       {time.perf_counter() - started:.1f} sec")
    print(f"Workbook:         {Path(cfg['output']).name}")
    return {"status": 0, "orders": orders, "itemsets": itemsets, "rules": rules, "reliable": good,
            "recommendations": recommendations, "bundles": bundles, "evaluation": evaluation}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args)
        if not Path(cfg["input"]).exists():
            raise ValueError(f"Input file not found: {cfg['input']}  (run: python make_sample_data.py)")
        return run(cfg, charts=not args.no_charts, lookup=args.lookup)["status"]
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
