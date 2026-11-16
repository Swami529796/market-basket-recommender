"""Tests for basket_recommender.py. Run with: python -m pytest"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import basket_recommender as br  # noqa: E402

CFG = {**br.DEFAULTS, "columns": dict(br.DEFAULTS["columns"])}


def baskets_of(*orders: str) -> list[frozenset[str]]:
    """'ABC' -> an order with items A, B and C."""
    return [frozenset(order) for order in orders]


def planted_baskets() -> list[frozenset[str]]:
    """1,000 orders. Printer is in 100 and Toner in 120, 60 of them together (a real link).
    Paper is in every second order whatever else is bought (a best-seller, no link)."""
    baskets = []
    for i in range(1000):
        items = {"Filler"}
        if i < 100:
            items.add("Printer")
        if i < 60 or 100 <= i < 160:
            items.add("Toner")
        if i % 2 == 0:
            items.add("Paper")
        baskets.append(frozenset(items))
    return baskets


def test_apriori_counts_a_small_example_by_hand():
    baskets = baskets_of("ABC", "AB", "AB", "AC", "BC", "D")
    found = br.frequent_itemsets(baskets, min_count=2, max_items=3)
    assert found == {("A",): 4, ("B",): 4, ("C",): 3,          # D is in one order only: dropped
                     ("A", "B"): 3, ("A", "C"): 2, ("B", "C"): 2}   # A+B+C is in one order: dropped
    assert br.frequent_itemsets(baskets, min_count=1, max_items=3)[("A", "B", "C")] == 1
    assert ("A", "B") not in br.frequent_itemsets(baskets, min_count=4, max_items=3)


def test_candidates_are_pruned_when_a_part_is_not_frequent():
    # A+B, A+C and B+C are all frequent, so A+B+C is worth counting
    assert br.generate_candidates({("A", "B"), ("A", "C"), ("B", "C")}, 3) == {("A", "B", "C")}
    # without B+C, the triple cannot be frequent and is never counted
    assert br.generate_candidates({("A", "B"), ("A", "C")}, 3) == set()


def test_rule_measures_match_the_readme_example():
    # 10 orders: Printer in 4, Toner in 5, both in 3
    m = br.rule_metrics(n=10, n_a=4, n_c=5, n_both=3)
    assert m["support"] == pytest.approx(0.30)
    assert m["confidence"] == pytest.approx(0.75)
    assert m["lift"] == pytest.approx(1.5)
    assert m["leverage"] == pytest.approx(0.30 - 0.4 * 0.5)
    assert m["conviction"] == pytest.approx((1 - 0.5) / (1 - 0.75))
    # Printer in 4, Paper in 8, both in 3: high confidence, yet below Paper's normal rate
    paper = br.rule_metrics(n=10, n_a=4, n_c=8, n_both=3)
    assert paper["confidence"] == pytest.approx(0.75) and paper["lift"] < 1
    assert br.rule_metrics(10, 2, 5, 2)["conviction"] == math.inf


def test_best_seller_is_not_reported_as_an_affinity():
    _, rules = br.mine(planted_baskets(), {**CFG, "min_orders": 20})
    good = br.reliable(rules)
    pairs = {(a, c) for a, c in zip(good["antecedent"], good["consequent"])}
    assert (("Printer",), "Toner") in pairs and (("Toner",), "Printer") in pairs
    assert "Paper" not in set(good["consequent"]) and "Filler" not in set(good["consequent"])

    printer_paper = rules[(rules["antecedent"] == ("Printer",)) & (rules["consequent"] == "Paper")].iloc[0]
    assert printer_paper["confidence"] == pytest.approx(0.5)      # looks strong ...
    assert printer_paper["lift"] == pytest.approx(1.0)            # ... but it is just Paper's normal rate
    assert not printer_paper["passes"]


def test_a_best_seller_riding_on_a_real_rule_is_set_aside():
    _, rules = br.mine(planted_baskets(), {**CFG, "min_orders": 20})
    rider = rules[(rules["antecedent"] == ("Paper", "Printer")) & (rules["consequent"] == "Toner")].iloc[0]
    assert rider["lift"] == pytest.approx(5.0)                    # lift alone would let it through
    assert rider["confidence_gain"] == pytest.approx(0.0)         # no better than Printer -> Toner
    assert rider["adds_nothing"] and not rider["passes"]


def test_significance_uses_fisher_for_small_counts_and_rejects_thin_evidence():
    p_large, test_large = br.significance(n=1000, n_a=100, n_c=120, n_both=60)
    assert test_large == "chi-square" and p_large < 1e-20
    p_small, test_small = br.significance(n=40, n_a=4, n_c=5, n_both=2)   # expected count 0.5
    assert test_small == "Fisher exact" and p_small > 0.05
    assert br.significance(n=10, n_a=10, n_c=5, n_both=5) == (1.0, "none")   # A is in every order


def test_bad_rows_are_reported_and_custom_column_names_work(tmp_path):
    file = tmp_path / "sales.csv"
    file.write_text("bill,sku,qty,rate\n"
                    "1,Pen,2,10\n"
                    "1,Ink,1,50\n"
                    "1,Pen,1,10\n"       # same product twice in the order
                    "2,Pen,-1,10\n"      # a return
                    "2,Ink,1,\n"         # no price
                    ",Pen,1,10\n"        # no order id
                    "3,,1,10\n"          # no product
                    "3,Pad,1,30\n", encoding="utf-8")
    columns = {"order": "bill", "product": "sku", "quantity": "qty", "price": "rate", "date": "d", "category": "c"}
    orders = br.load_orders(file, columns)
    assert orders.baskets == {"1": frozenset({"Pen", "Ink"}), "2": frozenset({"Ink"}), "3": frozenset({"Pad"})}
    reported = dict(zip(orders.issues["Problem"], orders.issues["Rows"]))
    assert reported == {"Order id is blank": 1, "Product is blank": 1,
                        "Quantity is zero or negative (return or cancellation)": 1,
                        "Price is missing or not positive": 1,
                        "Same product listed more than once in an order": 1}
    pen = orders.lines[(orders.lines["order"] == "1") & (orders.lines["item"] == "Pen")].iloc[0]
    assert pen["quantity"] == 3
    with pytest.raises(ValueError, match="Column 'order_id' not found in sales.csv"):
        br.load_orders(file, CFG["columns"])           # default names do not exist in this file


def test_recommender_ranks_by_confidence_and_never_suggests_what_is_in_the_basket():
    baskets = planted_baskets()
    _, rules = br.mine(baskets, {**CFG, "min_orders": 20})
    recommender = br.Recommender(br.reliable(rules), br.popularity_of(baskets))
    printer = frozenset({"Printer"})
    assert recommender.recommend(printer, 3, "rules") == ["Toner"]
    assert recommender.recommend(printer, 2, "best_sellers") == ["Filler", "Paper"]
    # blended: Toner (60% after a Printer) goes above Paper (50% of all orders), below Filler (100%)
    assert recommender.recommend(printer, 3, "blended") == ["Filler", "Toner", "Paper"]
    assert recommender.recommend(frozenset({"Printer", "Toner"}), 5, "rules") == []
    assert recommender.recommend(frozenset({"Paper"}), 5, "rules") == []      # no rule: nothing is invented


def test_sample_data_end_to_end(tmp_path, capsys):
    cfg = {**CFG, "input": str(ROOT / "data" / "order_lines.csv.gz"), "output": str(tmp_path / "out.xlsx"),
           "charts_dir": str(tmp_path)}
    result = br.run(cfg)
    assert "Strongest rules:" in capsys.readouterr().out
    good = result["reliable"]
    found = {(a[0], c) for a, c in zip(good["antecedent"], good["consequent"]) if len(a) == 1}

    # every strong and medium planted affinity is recovered
    planted = pd.read_csv(ROOT / "data" / "planted_affinities.csv")
    for _, row in planted[planted["strength"] != "weak"].iterrows():
        for add_on in row["add_on"].split(" + "):
            assert (row["trigger"], add_on) in found, f"missed {row['trigger']} -> {add_on}"
    triples = {tuple(sorted(a + (c,))) for a, c in zip(good["antecedent"], good["consequent"]) if len(a) == 2}
    assert ("Business Laptop", "Laptop Bag", "Wireless Mouse") in triples

    # the popular items co-occur with everything, yet no reliable rule involves them
    popular = {"A4 Copier Paper (Ream)", "Ball Pens (Box of 20)", "Pen Drive 64GB", "Notebook Pack",
               "Extension Board"}
    assert not popular & set(good["consequent"])
    assert not any(popular & set(a) for a in good["antecedent"])
    assert (good["orders_together"] >= cfg["min_orders"]).all() and (good["lift"] >= cfg["min_lift"]).all()

    # bundles: the arithmetic of the estimate holds and the laptop kit is offered
    bundles = result["bundles"]
    assert "Business Laptop + Laptop Bag + Wireless Mouse" in set(bundles["bundle"])
    assert (bundles["net_estimate"] - (bundles["extra_revenue"] - bundles["discount_given"])).abs().max() < 0.01
    assert (bundles["anchor_orders"] == bundles["already_complete"] + bundles["near_miss_orders"]).all()

    # evaluation: adding rules to the best-sellers must beat the best-sellers alone
    table = result["evaluation"]["table"].set_index("method")
    assert table.loc["Rules + best-sellers", "top_5"] > table.loc["Best-sellers (baseline)", "top_5"]
    assert result["evaluation"]["train_orders"] + result["evaluation"]["test_orders"] == len(result["orders"].baskets)

    workbook = load_workbook(tmp_path / "out.xlsx")
    assert workbook.sheetnames == ["Summary", "Rules", "Recommendations", "Bundles", "Evaluation", "Data Issues"]
    assert "ESTIMATE" in workbook["Bundles"]["A1"].value
    assert (tmp_path / "rules_lift_vs_support.png").exists() and (tmp_path / "hit_rate_vs_baseline.png").exists()
