# recycle
リサイクル

"""
TrashAlchemist-Sim
==================
廃プラスチック資源化・バイオリファイナリー総合シミュレーター

機能:
1. 原料処理・製品生産モデル
2. 収入・操業費・減価償却・営業利益の計算
3. ROI・損益分岐点・投資回収期間の算出
4. 微生物効率・製品価格・稼働率のシナリオ比較
5. 感度分析とグラフ出力
6. CSVへの計算結果保存
7. データの出所・信頼性区分の記録

重要:
このプログラムは事業性を検討するための計算モデルです。
初期設定値は説明用の仮定であり、実験結果・市場価格・
工場の実績を保証するものではありません。

実行:
    pip install pandas matplotlib
    python simulator.py
"""

from dataclasses import dataclass, asdict, replace
from typing import Optional
import math
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. パラメーター定義
# ============================================================

@dataclass
class SimulationConfig:
    """技術・経済パラメーター。金額は原則として日本円。"""

    # --- 原料・操業 ---
    annual_capacity_kg: float = 500_000
    utilization_rate: float = 1.0

    # 廃棄物を受け入れることで得る処理料金
    tipping_fee_yen_per_kg: float = 50.0

    # 原料の輸送・受入・管理費
    logistics_yen_per_kg: float = 20.0

    # 電力・熱・水などの費用
    energy_yen_per_kg: float = 0.0

    # 酵素・培養・薬品などの費用
    bioprocess_yen_per_kg: float = 0.0

    # 残渣の処分・追加処理費
    residue_yen_per_kg: float = 0.0

    # 年間固定操業費
    annual_fixed_opex_yen: float = 12_000_000

    # --- 製品 ---
    product_name: str = "Vanillin"

    # 販売可能な製品1kgあたりの想定価格
    product_price_yen_per_kg: float = 4_500.0

    # 基準となる理論上の製品収率
    # 原料プラスチック1kgあたりの製品kg
    base_yield_rate: float = 0.02

    # 分解・変換効率の倍率
    efficiency_multiplier: float = 1.0

    # 収率の上限。モデル上の仮定であり、
    # 物理的な普遍上限を示すものではない。
    maximum_yield_rate: float = 0.15

    # 生成した製品のうち回収・精製できる割合
    product_recovery_rate: float = 0.80

    # --- 精製 ---
    # 回収した製品1kgあたりの精製費
    purification_yen_per_kg_product: float = 500.0

    # --- 設備投資・会計 ---
    initial_capex_yen: float = 100_000_000
    asset_lifetime_years: int = 10
    annual_financing_cost_yen: float = 0.0
    annual_tax_rate: float = 0.0

    # --- データの管理 ---
    # illustrative_assumption / literature / measured
    data_status: str = "illustrative_assumption"


# ============================================================
# 2. 入力値の検証
# ============================================================

def validate_config(c: SimulationConfig) -> None:
    """不正なパラメーターによる計算を防止する。"""

    if c.annual_capacity_kg < 0:
        raise ValueError("年間処理能力は0以上にしてください。")

    if not 0 <= c.utilization_rate <= 1:
        raise ValueError("稼働率は0〜1の範囲にしてください。")

    if c.efficiency_multiplier <= 0:
        raise ValueError("効率倍率は0より大きくしてください。")

    if not 0 <= c.base_yield_rate <= 1:
        raise ValueError("基準収率は0〜1の範囲にしてください。")

    if not 0 <= c.maximum_yield_rate <= 1:
        raise ValueError("収率上限は0〜1の範囲にしてください。")

    if not 0 <= c.product_recovery_rate <= 1:
        raise ValueError("製品回収率は0〜1の範囲にしてください。")

    if c.asset_lifetime_years <= 0:
        raise ValueError("設備耐用年数は1年以上にしてください。")

    if c.initial_capex_yen < 0:
        raise ValueError("設備投資額は0以上にしてください。")

    if not 0 <= c.annual_tax_rate <= 1:
        raise ValueError("税率は0〜1の範囲にしてください。")

    numeric_values = [
        c.tipping_fee_yen_per_kg,
        c.logistics_yen_per_kg,
        c.energy_yen_per_kg,
        c.bioprocess_yen_per_kg,
        c.residue_yen_per_kg,
        c.annual_fixed_opex_yen,
        c.product_price_yen_per_kg,
        c.purification_yen_per_kg_product,
        c.annual_financing_cost_yen,
    ]

    if any(v < 0 for v in numeric_values):
        raise ValueError("費用・価格の設定値は0以上にしてください。")

    if c.data_status not in {
        "illustrative_assumption",
        "literature",
        "measured",
    }:
        raise ValueError(
            "data_status は illustrative_assumption / "
            "literature / measured のいずれかにしてください。"
        )


# ============================================================
# 3. 総合シミュレーター
# ============================================================

class TrashAlchemistSimulator:

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        validate_config(self.config)

    def run_simulation(self) -> dict:
        c = self.config

        # ----------------------------------------------------
        # A. 原料処理量
        # ----------------------------------------------------
        processed_kg = (
            c.annual_capacity_kg * c.utilization_rate
        )

        # ----------------------------------------------------
        # B. 製品収率
        # ----------------------------------------------------
        # 元コードの平方根モデルを維持。
        # 効率倍率と収率の関係は未検証のモデル仮定。
        gross_yield_rate = min(
            c.base_yield_rate * math.sqrt(
                c.efficiency_multiplier
            ),
            c.maximum_yield_rate,
        )

        gross_product_kg = processed_kg * gross_yield_rate

        # 実際に回収・精製できる製品量
        recovered_product_kg = (
            gross_product_kg * c.product_recovery_rate
        )

        # ----------------------------------------------------
        # C. 売上
        # ----------------------------------------------------
        tipping_revenue = (
            processed_kg * c.tipping_fee_yen_per_kg
        )

        product_revenue = (
            recovered_product_kg
            * c.product_price_yen_per_kg
        )

        total_revenue = tipping_revenue + product_revenue

        # ----------------------------------------------------
        # D. 変動費
        # ----------------------------------------------------
        logistics_cost = (
            processed_kg * c.logistics_yen_per_kg
        )

        energy_cost = (
            processed_kg * c.energy_yen_per_kg
        )

        bioprocess_cost = (
            processed_kg * c.bioprocess_yen_per_kg
        )

        residue_cost = (
            processed_kg * c.residue_yen_per_kg
        )

        purification_cost = (
            recovered_product_kg
            * c.purification_yen_per_kg_product
        )

        total_variable_cost = sum([
            logistics_cost,
            energy_cost,
            bioprocess_cost,
            residue_cost,
            purification_cost,
        ])

        # ----------------------------------------------------
        # E. 営業損益・キャッシュフロー
        # ----------------------------------------------------
        total_opex = (
            c.annual_fixed_opex_yen + total_variable_cost
        )

        # EBITDA: 減価償却前の営業利益相当
        ebitda = total_revenue - total_opex

        # 定額法による簡易減価償却
        annual_depreciation = (
            c.initial_capex_yen / c.asset_lifetime_years
        )

        # EBIT: 支払利息等を控除する前の営業利益
        ebit = ebitda - annual_depreciation

        # 税額は、簡易モデルとして正の課税所得にのみ適用。
        # 実際の税務計算・欠損金・控除等は未実装。
        taxable_profit = max(
            0.0,
            ebit - c.annual_financing_cost_yen,
        )

        estimated_tax = taxable_profit * c.annual_tax_rate

        estimated_net_profit = (
            ebit
            - c.annual_financing_cost_yen
            - estimated_tax
        )

        # 簡易的な年間営業キャッシュフロー。
        # 借入元本返済・運転資本変動・追加投資等は含まない。
        annual_cash_flow = (
            estimated_net_profit + annual_depreciation
        )

        # ----------------------------------------------------
        # F. 投資指標
        # ----------------------------------------------------
        roi_on_capex = (
            ebit / c.initial_capex_yen * 100
            if c.initial_capex_yen > 0
            else None
        )

        payback_years = (
            c.initial_capex_yen / annual_cash_flow
            if annual_cash_flow > 0
            else None
        )

        # 変動費を含む1kgあたりの限界利益
        contribution_per_kg = (
            c.tipping_fee_yen_per_kg
            + gross_yield_rate
            * c.product_recovery_rate
            * c.product_price_yen_per_kg
            - c.logistics_yen_per_kg
            - c.energy_yen_per_kg
            - c.bioprocess_yen_per_kg
            - c.residue_yen_per_kg
            - gross_yield_rate
            * c.product_recovery_rate
            * c.purification_yen_per_kg_product
        )

        # 固定費と減価償却をカバーするための処理量
        break_even_kg = (
            (
                c.annual_fixed_opex_yen
                + annual_depreciation
            ) / contribution_per_kg
            if contribution_per_kg > 0
            else None
        )

        break_even_reachable = (
            break_even_kg is not None
            and break_even_kg <= c.annual_capacity_kg
        )

        # ----------------------------------------------------
        # G. 結果
        # ----------------------------------------------------
        return {
            "Data Status": c.data_status,
            "Product": c.product_name,
            "Efficiency Multiplier": c.efficiency_multiplier,
            "Annual Capacity (kg)": c.annual_capacity_kg,
            "Utilization Rate (%)": c.utilization_rate * 100,
            "Processed Feedstock (kg)": processed_kg,
            "Gross Yield Rate (%)": gross_yield_rate * 100,
            "Product Recovery Rate (%)": (
                c.product_recovery_rate * 100
            ),
            "Gross Product (kg)": gross_product_kg,
            "Recovered Product (kg)": recovered_product_kg,
            "Tipping Revenue (JPY)": tipping_revenue,
            "Product Revenue (JPY)": product_revenue,
            "Total Revenue (JPY)": total_revenue,
            "Logistics Cost (JPY)": logistics_cost,
            "Energy Cost (JPY)": energy_cost,
            "Bioprocess Cost (JPY)": bioprocess_cost,
            "Residue Cost (JPY)": residue_cost,
            "Purification Cost (JPY)": purification_cost,
            "Fixed OPEX (JPY)": c.annual_fixed_opex_yen,
            "Total OPEX (JPY)": total_opex,
            "EBITDA (JPY)": ebitda,
            "Depreciation (JPY)": annual_depreciation,
            "EBIT (JPY)": ebit,
            "Financing Cost (JPY)": c.annual_financing_cost_yen,
            "Estimated Tax (JPY)": estimated_tax,
            "Estimated Net Profit (JPY)": estimated_net_profit,
            "Annual Cash Flow (JPY)": annual_cash_flow,
            "ROI on CAPEX (%)": roi_on_capex,
            "Payback Period (years)": payback_years,
            "Contribution per kg (JPY)": contribution_per_kg,
            "Break-even Throughput (kg)": break_even_kg,
            "Break-even Reachable": break_even_reachable,
        }


# ============================================================
# 4. シナリオ比較
# ============================================================

def run_scenario_analysis(
    base_config: SimulationConfig,
    efficiency_values=(1, 2, 3, 5, 7, 10),
) -> pd.DataFrame:
    """微生物効率の仮定を変えて結果を比較する。"""

    results = []

    for efficiency in efficiency_values:
        scenario_config = replace(
            base_config,
            efficiency_multiplier=float(efficiency),
        )

        result = TrashAlchemistSimulator(
            scenario_config
        ).run_simulation()

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# 5. 感度分析
# ============================================================

def run_price_sensitivity(
    base_config: SimulationConfig,
    price_values=(1000, 2500, 4500, 6000, 8000),
) -> pd.DataFrame:
    """製品販売価格が収益性に与える影響を確認する。"""

    results = []

    for price in price_values:
        scenario_config = replace(
            base_config,
            product_price_yen_per_kg=float(price),
        )

        result = TrashAlchemistSimulator(
            scenario_config
        ).run_simulation()

        results.append({
            "Product Price (JPY/kg)": price,
            "Product Output (kg)": result["Recovered Product (kg)"],
            "Total Revenue (JPY)": result["Total Revenue (JPY)"],
            "EBITDA (JPY)": result["EBITDA (JPY)"],
            "EBIT (JPY)": result["EBIT (JPY)"],
            "ROI on CAPEX (%)": result["ROI on CAPEX (%)"],
        })

    return pd.DataFrame(results)


def run_utilization_sensitivity(
    base_config: SimulationConfig,
    utilization_values=(0.3, 0.5, 0.7, 0.85, 1.0),
) -> pd.DataFrame:
    """設備稼働率が利益に与える影響を確認する。"""

    results = []

    for utilization in utilization_values:
        scenario_config = replace(
            base_config,
            utilization_rate=float(utilization),
        )

        result = TrashAlchemistSimulator(
            scenario_config
        ).run_simulation()

        results.append({
            "Utilization (%)": utilization * 100,
            "Processed Feedstock (kg)": result[
                "Processed Feedstock (kg)"
            ],
            "Recovered Product (kg)": result[
                "Recovered Product (kg)"
            ],
            "EBITDA (JPY)": result["EBITDA (JPY)"],
            "EBIT (JPY)": result["EBIT (JPY)"],
        })

    return pd.DataFrame(results)


# ============================================================
# 6. 結果の保存
# ============================================================

def save_results(
    scenarios: pd.DataFrame,
    price_sensitivity: pd.DataFrame,
    utilization_sensitivity: pd.DataFrame,
) -> None:
    scenarios.to_csv(
        "scenario_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    price_sensitivity.to_csv(
        "price_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    utilization_sensitivity.to_csv(
        "utilization_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 7. グラフ
# ============================================================

def create_charts(
    scenarios: pd.DataFrame,
    price_sensitivity: pd.DataFrame,
    utilization_sensitivity: pd.DataFrame,
) -> None:

    # グラフ1: 微生物効率と営業利益
    plt.figure(figsize=(9, 5))
    plt.plot(
        scenarios["Efficiency Multiplier"],
        scenarios["EBIT (JPY)"] / 1_000_000,
        marker="o",
        linewidth=2,
    )
    plt.title("Operating Profit vs. Efficiency Assumption")
    plt.xlabel("Efficiency Multiplier")
    plt.ylabel("EBIT (Million JPY/year)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("efficiency_vs_profit.png", dpi=150)
    plt.close()

    # グラフ2: 製品価格と営業利益
    plt.figure(figsize=(9, 5))
    plt.plot(
        price_sensitivity["Product Price (JPY/kg)"],
        price_sensitivity["EBIT (JPY)"] / 1_000_000,
        marker="o",
        linewidth=2,
    )
    plt.title("Product Price Sensitivity")
    plt.xlabel("Product Price (JPY/kg)")
    plt.ylabel("EBIT (Million JPY/year)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("price_sensitivity.png", dpi=150)
    plt.close()

    # グラフ3: 稼働率と営業利益
    plt.figure(figsize=(9, 5))
    plt.plot(
        utilization_sensitivity["Utilization (%)"],
        utilization_sensitivity["EBIT (JPY)"] / 1_000_000,
        marker="o",
        linewidth=2,
    )
    plt.title("Operating Profit vs. Plant Utilization")
    plt.xlabel("Utilization (%)")
    plt.ylabel("EBIT (Million JPY/year)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("utilization_sensitivity.png", dpi=150)
    plt.close()


# ============================================================
# 8. 実行
# ============================================================

def main():
    print("=" * 65)
    print("TrashAlchemist-Sim: Integrated Biorefinery Simulator")
    print("=" * 65)

    # 初期設定は説明用の仮定。
    # 実証済みの収率・市場価格・工場コストではない。
    config = SimulationConfig(
        annual_capacity_kg=500_000,
        utilization_rate=1.0,
        tipping_fee_yen_per_kg=50,
        logistics_yen_per_kg=20,
        energy_yen_per_kg=0,
        bioprocess_yen_per_kg=0,
        residue_yen_per_kg=0,
        annual_fixed_opex_yen=12_000_000,
        product_name="Vanillin",
        product_price_yen_per_kg=4_500,
        base_yield_rate=0.02,
        efficiency_multiplier=1.0,
        maximum_yield_rate=0.15,
        product_recovery_rate=0.80,
        purification_yen_per_kg_product=500,
        initial_capex_yen=100_000_000,
        asset_lifetime_years=10,
        annual_financing_cost_yen=0,
        annual_tax_rate=0.0,
        data_status="illustrative_assumption",
    )

    # 基本シナリオ
    base_result = TrashAlchemistSimulator(
        config
    ).run_simulation()

    print("\n[BASE CASE]")
    print(f"Data status: {config.data_status}")
    print(f"Product: {config.product_name}")
    print(
        "Recovered product: "
        f"{base_result['Recovered Product (kg)']:,.2f} kg/year"
    )
    print(
        "Total revenue: "
        f"¥{base_result['Total Revenue (JPY)']:,.0f}"
    )
    print(
        "Total operating expenses: "
        f"¥{base_result['Total OPEX (JPY)']:,.0f}"
    )
    print(
        "EBITDA: "
        f"¥{base_result['EBITDA (JPY)']:,.0f}"
    )
    print(
        "EBIT: "
        f"¥{base_result['EBIT (JPY)']:,.0f}"
    )

    if base_result["ROI on CAPEX (%)"] is not None:
        print(
            "ROI on CAPEX: "
            f"{base_result['ROI on CAPEX (%)']:.2f}%"
        )

    if base_result["Payback Period (years)"] is not None:
        print(
            "Simple payback period: "
            f"{base_result['Payback Period (years)']:.2f} years"
        )
    else:
        print("Simple payback period: Not available (cash flow <= 0)")

    if base_result["Break-even Throughput (kg)"] is not None:
        print(
            "Break-even throughput: "
            f"{base_result['Break-even Throughput (kg)']:,.0f} kg/year"
        )
        print(
            "Break-even reachable at nominal capacity: "
            f"{base_result['Break-even Reachable']}"
        )
    else:
        print(
            "Break-even throughput: Not available "
            "(non-positive contribution per kg)"
        )

    # 効率倍率のシナリオ比較
    scenarios = run_scenario_analysis(config)

    print("\n[EFFICIENCY SCENARIOS]")
    print(
        scenarios[
            [
                "Efficiency Multiplier",
                "Recovered Product (kg)",
                "Total Revenue (JPY)",
                "EBITDA (JPY)",
                "EBIT (JPY)",
                "ROI on CAPEX (%)",
            ]
        ].round(2).to_string(index=False)
    )

    # 製品価格の感度分析
    prices = run_price_sensitivity(config)

    print("\n[PRICE SENSITIVITY]")
    print(prices.round(2).to_string(index=False))

    # 稼働率の感度分析
    utilization = run_utilization_sensitivity(config)

    print("\n[UTILIZATION SENSITIVITY]")
    print(utilization.round(2).to_string(index=False))

    # 保存
    save_results(scenarios, prices, utilization)

    create_charts(scenarios, prices, utilization)

    # 設定値とデータ区分を保存
    with open(
        "simulation_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(config),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[OUTPUT FILES]")
    print("- scenario_results.csv")
    print("- price_sensitivity.csv")
    print("- utilization_sensitivity.csv")
    print("- efficiency_vs_profit.png")
    print("- price_sensitivity.png")
    print("- utilization_sensitivity.png")
    print("- simulation_config.json")

    print("\n[IMPORTANT]")
    print(
        "All default technical and economic values are illustrative "
        "assumptions. Validate them against experiments, literature, "
        "supplier quotations, and actual operating data before making "
        "investment decisions."
    )


if __name__ == "__main__":
    main()
