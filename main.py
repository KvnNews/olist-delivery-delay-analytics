from __future__ import annotations

import os
from pathlib import Path
import textwrap
import json

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

EXPECTED_FILES = set(DATA_FILES.values())


def find_dataset_directory() -> Path | None:
    candidates = [Path.cwd(), Path.cwd() / "dataset", Path.cwd() / "data", Path.cwd() / "data-set"]
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in EXPECTED_FILES):
            return candidate
    return None


def load_data(base_path: Path) -> dict[str, pd.DataFrame]:
    parsers = {
        "orders": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "items": ["shipping_limit_date"],
        "reviews": ["review_creation_date", "review_answer_timestamp"],
    }
    data = {}
    for key, filename in DATA_FILES.items():
        path = base_path / filename
        parse_dates = parsers.get(key, [])
        data[key] = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    return data


def aggregate_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    return (
        geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "median"),
            geolocation_lng=("geolocation_lng", "median"),
        )
    )


def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lng / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def prepare_order_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    orders["order_purchase_date"] = orders["order_purchase_timestamp"].dt.date

    customers = data["customers"].copy()
    sellers = data["sellers"].copy()
    payments = data["payments"].copy()
    reviews = data["reviews"].copy()
    products = data["products"].copy()
    categories = data["categories"].copy()
    geo = aggregate_geolocation(data["geolocation"])

    products = products.merge(categories, on="product_category_name", how="left")
    items = data["items"].copy().merge(products, on="product_id", how="left")

    payments_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_types=("payment_type", lambda x: ",".join(sorted(set(x.dropna())))),
        )
    )

    items_agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            total_items=("order_item_id", "count"),
            revenue=("price", "sum"),
            freight_value=("freight_value", "sum"),
            avg_item_price=("price", "mean"),
            avg_product_weight=("product_weight_g", "mean"),
            avg_length_cm=("product_length_cm", "mean"),
            avg_height_cm=("product_height_cm", "mean"),
            avg_width_cm=("product_width_cm", "mean"),
            distinct_categories=("product_category_name_english", "nunique"),
            category_most_common=("product_category_name_english", lambda x: x.mode().iloc[0] if len(x.dropna()) else "unknown"),
        )
    )

    order_seller = (
        items.groupby("order_id", as_index=False)
        .agg(
            seller_id=("seller_id", lambda x: x.mode().iloc[0] if len(x.dropna()) else np.nan),
            seller_count=("seller_id", "nunique"),
        )
    )
    items_agg = items_agg.merge(order_seller, on="order_id", how="left")

    customer_geo = customers.merge(
        geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"}
    )
    seller_geo = sellers.merge(
        geo,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"}
    )

    order_base = (
        orders.merge(customer_geo, on="customer_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
        .merge(items_agg, on="order_id", how="left")
    )

    seller_geo_unique = seller_geo[["seller_id", "seller_state", "seller_lat", "seller_lng"]].drop_duplicates("seller_id")
    order_base = order_base.merge(seller_geo_unique, on="seller_id", how="left")

    order_base["estimated_delivery_days"] = (
        order_base["order_estimated_delivery_date"] - order_base["order_purchase_timestamp"]
    ).dt.days
    order_base["actual_delivery_days"] = (
        order_base["order_delivered_customer_date"] - order_base["order_purchase_timestamp"]
    ).dt.days
    order_base["delivery_delay"] = (
        order_base["order_delivered_customer_date"] - order_base["order_estimated_delivery_date"]
    ).dt.days
    order_base["is_late"] = order_base["delivery_delay"] > 0
    order_base["freight_ratio"] = order_base["freight_value"] / order_base["revenue"].replace({0: np.nan})
    order_base["distance_km"] = haversine_distance(
        order_base["customer_lat"].fillna(0),
        order_base["customer_lng"].fillna(0),
        order_base["seller_lat"].fillna(0),
        order_base["seller_lng"].fillna(0),
    )

    order_base["purchase_weekday"] = order_base["order_purchase_timestamp"].dt.weekday
    order_base["purchase_month"] = order_base["order_purchase_timestamp"].dt.month
    order_base["payment_type_primary"] = order_base["payment_types"].str.split(",").str[0]

    seller_delay = (
        order_base.groupby("seller_id", as_index=False)
        .agg(seller_orders=("order_id", "count"), seller_late_rate=("is_late", "mean"))
    )
    customer_history = (
        order_base.groupby("customer_id", as_index=False)
        .agg(customer_orders=("order_id", "count"), customer_late_rate=("is_late", "mean"))
    )
    order_base = order_base.merge(seller_delay, on="seller_id", how="left").merge(customer_history, on="customer_id", how="left")

    order_base["review_score"] = order_base["review_score"].fillna(order_base["review_score"].median())
    order_base["payment_installments"] = order_base["payment_installments"].fillna(1)

    return order_base


def business_diagnostics(order_base: pd.DataFrame) -> dict[str, object]:
    delivered = order_base[order_base["order_status"] == "delivered"].copy()
    total_revenue = delivered["revenue"].sum()
    revenue_by_category = (
        delivered.groupby("category_most_common", as_index=False)
        .agg(revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
    )
    revenue_by_state = (
        delivered.groupby("customer_state", as_index=False)
        .agg(revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
    )
    delay_by_state = (
        delivered.groupby("customer_state", as_index=False)
        .agg(
            late_rate=("is_late", "mean"),
            median_delay=("delivery_delay", "median"),
            revenue=("revenue", "sum"),
        )
        .sort_values(["late_rate", "median_delay"], ascending=False)
    )
    metrics = {
        "total_revenue": total_revenue,
        "top_category": revenue_by_category.iloc[0].to_dict() if not revenue_by_category.empty else {},
        "top_state": revenue_by_state.iloc[0].to_dict() if not revenue_by_state.empty else {},
        "delay_state_rank": delay_by_state.head(10).to_dict(orient="records"),
        "avg_ticket": delivered["revenue"].mean(),
        "avg_delivery_days": delivered["actual_delivery_days"].mean(),
        "late_rate": delivered["is_late"].mean(),
        "avg_review_score": delivered["review_score"].mean(),
        "revenue_by_category": revenue_by_category,
        "revenue_by_state": revenue_by_state,
        "delay_by_state": delay_by_state,
    }
    return metrics


def build_dashboard(metrics: dict[str, object], order_base: pd.DataFrame, monitoring_note: str = "") -> None:
    delivered = order_base[order_base["order_status"] == "delivered"].copy()
    revenue_time = (
        delivered.groupby("order_purchase_date", as_index=False).agg(revenue=("revenue", "sum"))
    )
    delivery_month = (
        delivered.groupby("purchase_month", as_index=False)
        .agg(late_rate=("is_late", "mean"), revenue=("revenue", "sum"))
    )

    all_states = sorted(delivered["customer_state"].dropna().unique())
    all_categories = sorted(delivered["category_most_common"].dropna().unique())
    min_date = delivered["order_purchase_date"].min().isoformat()
    max_date = delivered["order_purchase_date"].max().isoformat()

    filter_df = (
        delivered.groupby(
            ["order_purchase_date", "customer_state", "category_most_common", "purchase_month"],
            as_index=False,
        )
        .agg(
            sum_revenue=("revenue", "sum"),
            sum_late=("is_late", "sum"),
            sum_delivery_days=("actual_delivery_days", "sum"),
            sum_review_score=("review_score", "sum"),
            order_count=("order_id", "count"),
        )
    )
    filter_df["order_purchase_date"] = filter_df["order_purchase_date"].astype(str)
    filter_data = json.dumps(filter_df.to_dict(orient="records"), default=str)

    state_options = "".join(
        [f"<option value='{state}'>{state}</option>" for state in all_states]
    )
    category_options = "".join(
        [f"<option value='{category}'>{category}</option>" for category in all_categories]
    )

    fig_revenue_time = px.line(
        revenue_time,
        x="order_purchase_date",
        y="revenue",
        title="Receita ao longo do tempo",
        template="plotly_white",
    )
    fig_revenue_time.update_layout(hovermode="x unified", margin=dict(t=40, b=40, l=40, r=40), autosize=True)

    fig_state = px.bar(
        metrics["revenue_by_state"].head(12),
        x="customer_state",
        y="revenue",
        title="Top 12 estados por receita",
        template="plotly_white",
        color="revenue",
        color_continuous_scale="Blues",
    )
    fig_state.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=40, l=40, r=40), autosize=True)

    fig_category = px.bar(
        metrics["revenue_by_category"].head(12),
        x="category_most_common",
        y="revenue",
        title="Top 12 categorias por receita",
        template="plotly_white",
        color="revenue",
        color_continuous_scale="Purples",
    )
    fig_category.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False, margin=dict(t=40, b=80, l=40, r=40), autosize=True)

    fig_delay_state = px.bar(
        metrics["delay_by_state"].head(12),
        x="customer_state",
        y="late_rate",
        title="Top 12 estados com maior taxa de atraso",
        template="plotly_white",
        color="late_rate",
        color_continuous_scale="OrRd",
    )
    fig_delay_state.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%", margin=dict(t=40, b=40, l=40, r=40), autosize=True)

    fig_delivery_time = px.histogram(
        delivered,
        x="actual_delivery_days",
        nbins=30,
        title="Distribuição de dias de entrega",
        template="plotly_white",
    )
    fig_delivery_time.update_layout(margin=dict(t=40, b=40, l=40, r=40), autosize=True)

    fig_review = px.histogram(
        delivered,
        x="review_score",
        nbins=5,
        title="Distribuição de notas de avaliação",
        template="plotly_white",
    )
    fig_review.update_layout(xaxis=dict(dtick=1), margin=dict(t=40, b=40, l=40, r=40), autosize=True)

    fig_purchase_month = px.line(
        delivery_month,
        x="purchase_month",
        y="late_rate",
        title="Taxa de atraso por mês de compra",
        template="plotly_white",
    )
    fig_purchase_month.update_layout(yaxis_tickformat=".0%", margin=dict(t=40, b=40, l=40, r=40), autosize=True)

    kpis = {
        "Receita Total": f"R$ {metrics['total_revenue']:,.2f}",
        "Ticket Médio": f"R$ {metrics['avg_ticket']:,.2f}",
        "Tempo Médio de Entrega": f"{metrics['avg_delivery_days']:.1f} dias",
        "Taxa de Atraso": f"{metrics['late_rate']:.2%}",
        "Avaliação Média": f"{metrics['avg_review_score']:.2f}",
    }

    top_delay = metrics["delay_state_rank"][0] if metrics["delay_state_rank"] else {"customer_state": "N/A", "late_rate": 0}
    summary_cards = f"""
        <div class='kpi-grid'>
            {''.join([f"<div class='kpi-card'><span class='kpi-label'>{label}</span><span class='kpi-value'>{value}</span></div>" for label, value in kpis.items()])}
        </div>
        <div class='insights'>
          <div class='insight-block'>
            <h3>Categoria líder</h3>
            <p>{metrics['top_category']['category_most_common']} com R$ {metrics['top_category']['revenue']:,.2f} em receita.</p>
          </div>
          <div class='insight-block'>
            <h3>Estado principal</h3>
            <p>{metrics['top_state']['customer_state']} gerou R$ {metrics['top_state']['revenue']:,.2f}.</p>
          </div>
          <div class='insight-block'>
            <h3>Maior atraso</h3>
            <p>{top_delay['customer_state']} com taxa de atraso de {top_delay['late_rate']:.2%}.</p>
          </div>
        </div>
    """

    filters_html = f"""
      <div class='filter-panel'>
        <div class='filter-group'>
          <label>Estado
            <select id='stateFilter'><option value='all'>Todos</option>{state_options}</select>
          </label>
          <label>Categoria
            <select id='categoryFilter'><option value='all'>Todas</option>{category_options}</select>
          </label>
          <label>Período
            <input type='date' id='dateFrom' value='{min_date}' min='{min_date}' max='{max_date}' />
            <input type='date' id='dateTo' value='{max_date}' min='{min_date}' max='{max_date}' />
          </label>
        </div>
        <div class='filter-actions'>
          <button id='applyFilters'>Aplicar filtros</button>
          <button id='resetFilters'>Limpar</button>
        </div>
      </div>
    """

    delay_rows = "".join(
        [
            f"<tr><td>{row['customer_state']}</td><td>{row['revenue']:,.2f}</td><td>{row['late_rate']:.2%}</td><td>{row['median_delay']:.1f}</td></tr>"
            for row in metrics["delay_state_rank"]
        ]
    )
    delay_table = f"""
        <div class='table-card'>
          <h3>Estados com mais atrasos</h3>
          <table>
            <thead><tr><th>Estado</th><th>Receita</th><th>Taxa de atraso</th><th>Mediana atraso</th></tr></thead>
            <tbody>{delay_rows}</tbody>
          </table>
        </div>
    """

    page_style = """
      html, body { width: 100%; min-height: 100%; margin: 0; padding: 0; }
      body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f3f5f9; color: #1f2a37; }
      .page { width: 100vw; max-width: none; margin: 0; padding: 24px 32px; box-sizing: border-box; }
      .hero { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
      .hero h1 { margin: 0; font-size: 2.6rem; letter-spacing: -0.03em; }
      .hero p { margin: 8px 0 0; color: #5f7285; max-width: 940px; line-height: 1.6; }
      .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 24px 0; }
      .kpi-card { background: #ffffff; padding: 24px; border-radius: 20px; box-shadow: 0 14px 40px rgba(31, 42, 55, 0.08); min-height: 120px; display: flex; flex-direction: column; justify-content: center; }
      .kpi-label { color: #5f7285; font-size: 0.95rem; }
      .kpi-value { margin-top: 8px; font-size: 1.65rem; font-weight: 700; color: #0f172a; }
      .insights { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
      .insight-block { background: linear-gradient(135deg, #ffffff 0%, #f7f9fc 100%); padding: 22px 24px; border-radius: 20px; border: 1px solid rgba(15, 23, 42, 0.06); }
      .insight-block h3 { margin: 0 0 10px; font-size: 1.05rem; color: #0f172a; }
      .insight-block p { margin: 0; color: #475569; line-height: 1.7; }
      .charts { display: grid; gap: 24px; width: 100%; }
      .chart-card, .table-card, .kpi-card, .insight-block { width: 100%; }
      .chart-card { background: #ffffff; border-radius: 22px; padding: 22px; box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08); }
      .chart-card h2 { margin: 0 0 16px; font-size: 1.1rem; }
      .table-card { background: #ffffff; border-radius: 22px; padding: 22px; box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08); margin-top: 24px; }
      table { width: 100%; border-collapse: collapse; margin-top: 12px; }
      th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }
      th { background: #f8fafc; color: #334155; font-weight: 700; }
      tbody tr:hover { background: #f8fafc; }
      .grid-two { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
      .plotly-graph-div, .plotly-graph-div svg, .plotly-graph-div .main-svg { width: 100% !important; }
      .js-plotly-plot, .js-plotly-plot > div { width: 100% !important; }
      .filter-panel { background: #ffffff; padding: 20px; border-radius: 20px; box-shadow: 0 14px 40px rgba(31, 42, 55, 0.08); margin-bottom: 24px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; }
      .filter-group { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; width: 100%; }
      .filter-group label { display: flex; flex-direction: column; gap: 8px; font-weight: 600; color: #334155; }
      .filter-group select, .filter-group input { width: 100%; padding: 12px 12px; border: 1px solid #d2d6dc; border-radius: 14px; background: #f8fafc; color: #0f172a; }
      .filter-actions { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; justify-content: flex-end; }
      .filter-actions button { border: none; padding: 12px 24px; border-radius: 14px; font-weight: 700; cursor: pointer; background: #2563eb; color: white; box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18); transition: transform 0.18s ease, background 0.18s ease; }
      .filter-actions button:hover { transform: translateY(-1px); background: #1d4ed8; }
      .monitoring-card { background: #ffffff; border-radius: 22px; padding: 22px; box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08); margin-top: 24px; }
      .monitoring-card h3 { margin: 0 0 12px; font-size: 1.1rem; }
      .grid-two { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
      @media(max-width: 1200px) { .grid-two { grid-template-columns: 1fr; } }
      @media(max-width: 960px) { .kpi-grid, .insights { grid-template-columns: 1fr; } }
    """

    monitoring_html = (
        f"<div class='monitoring-card'><h3>Monitoramento</h3><p>{monitoring_note}</p></div>"
        if monitoring_note
        else ""
    )

    dashboard_path = Path("dashboard.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(
            f"<html><head><meta charset='utf-8'><title>Painel Olist</title><style>{page_style}</style><script src='https://cdn.plot.ly/plotly-latest.min.js'></script></head><body>"
        )
        f.write("<div class='page'>")
        f.write("<div class='hero'>")
        f.write("<div><h1>Painel Executivo Olist</h1><p>Uma visão moderna e executiva do desempenho de vendas, atraso de entregas e experiência do cliente para o dataset brasileiro da Olist.</p></div>")
        f.write("</div>")
        f.write(filters_html)
        f.write(summary_cards)
        f.write(monitoring_html)
        f.write("<div class='charts'>")
        f.write("<div class='chart-card'>" + fig_revenue_time.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='revenue_time_chart') + "</div>")
        f.write("<div class='grid-two'>")
        f.write("<div class='chart-card'>" + fig_state.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='state_revenue_chart') + "</div>")
        f.write("<div class='chart-card'>" + fig_category.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='category_revenue_chart') + "</div>")
        f.write("</div>")
        f.write("<div class='grid-two'>")
        f.write("<div class='chart-card'>" + fig_delay_state.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='delay_state_chart') + "</div>")
        f.write("<div class='chart-card'>" + fig_purchase_month.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='purchase_month_chart') + "</div>")
        f.write("</div>")
        f.write("<div class='grid-two'>")
        f.write("<div class='chart-card'>" + fig_delivery_time.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='delivery_time_chart') + "</div>")
        f.write("<div class='chart-card'>" + fig_review.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True}, div_id='review_score_chart') + "</div>")
        f.write("</div>")
        f.write(delay_table)
        f.write("</div>")
        f.write("</div>")
        f.write("<script>\n")
        f.write("const dashboardFilterData = " + filter_data + ";\n")
        f.write("const stateFilter = document.getElementById('stateFilter');\n")
        f.write("const categoryFilter = document.getElementById('categoryFilter');\n")
        f.write("const dateFrom = document.getElementById('dateFrom');\n")
        f.write("const dateTo = document.getElementById('dateTo');\n")
        f.write("const applyFiltersButton = document.getElementById('applyFilters');\n")
        f.write("const resetFiltersButton = document.getElementById('resetFilters');\n")
        f.write("const totalRevenueCard = document.querySelector(\".kpi-card:nth-child(1) .kpi-value\");\n")
        f.write("const avgTicketCard = document.querySelector(\".kpi-card:nth-child(2) .kpi-value\");\n")
        f.write("const avgDeliveryCard = document.querySelector(\".kpi-card:nth-child(3) .kpi-value\");\n")
        f.write("const lateRateCard = document.querySelector(\".kpi-card:nth-child(4) .kpi-value\");\n")
        f.write("const avgReviewCard = document.querySelector(\".kpi-card:nth-child(5) .kpi-value\");\n")
        f.write("function formatCurrency(value) { return new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(value); }\n")
        f.write("function formatPercent(value) { return (value * 100).toFixed(2) + '%'; }\n")
        f.write("function aggregate(data, groupKeys, sumKeys) { const aggregated = {}; data.forEach(row => { const key = groupKeys.map(k => row[k]).join('|'); if (!aggregated[key]) { aggregated[key] = { }; groupKeys.forEach(k => { aggregated[key][k] = row[k]; }); sumKeys.forEach(s => { aggregated[key][s] = 0; }); } sumKeys.forEach(s => { aggregated[key][s] += Number(row[s] || 0); }); }); return Object.values(aggregated); }\n")
        f.write("function applyAllFilters() { const filtered = dashboardFilterData.filter(row => { const stateMatch = stateFilter.value === 'all' || row.customer_state === stateFilter.value; const categoryMatch = categoryFilter.value === 'all' || row.category_most_common === categoryFilter.value; const afterStart = !dateFrom.value || row.order_purchase_date >= dateFrom.value; const beforeEnd = !dateTo.value || row.order_purchase_date <= dateTo.value; return stateMatch && categoryMatch && afterStart && beforeEnd; }); updateDashboard(filtered); }\n")
        f.write("function updateDashboard(data) { const totals = data.reduce((acc, row) => { acc.revenue += row.sum_revenue; acc.late += row.sum_late; acc.delivery += row.sum_delivery_days; acc.review += row.sum_review_score; acc.orders += row.order_count; return acc; }, { revenue: 0, late: 0, delivery: 0, review: 0, orders: 0 }); totalRevenueCard.textContent = formatCurrency(totals.revenue); avgTicketCard.textContent = totals.orders ? formatCurrency(totals.revenue / totals.orders) : '-'; avgDeliveryCard.textContent = totals.orders ? (totals.delivery / totals.orders).toFixed(1) + ' dias' : '-'; lateRateCard.textContent = totals.orders ? formatPercent(totals.late / totals.orders) : '-'; avgReviewCard.textContent = totals.orders ? (totals.review / totals.orders).toFixed(2) : '-'; updateCharts(data); }\n")
        f.write("function updateCharts(data) { const revenueByDate = aggregate(data, ['order_purchase_date'], ['sum_revenue']); revenueByDate.sort((a, b) => a.order_purchase_date.localeCompare(b.order_purchase_date)); const stateRevenue = aggregate(data, ['customer_state'], ['sum_revenue']); stateRevenue.sort((a, b) => b.sum_revenue - a.sum_revenue); const categoryRevenue = aggregate(data, ['category_most_common'], ['sum_revenue']); categoryRevenue.sort((a, b) => b.sum_revenue - a.sum_revenue); const delayByState = aggregate(data, ['customer_state'], ['sum_late', 'order_count']); delayByState.forEach(row => { row.late_rate = row.order_count ? row.sum_late / row.order_count : 0; }); delayByState.sort((a, b) => b.late_rate - a.late_rate); const monthlyDelay = aggregate(data, ['purchase_month'], ['sum_late', 'order_count']); monthlyDelay.sort((a, b) => Number(a.purchase_month) - Number(b.purchase_month)); const revenueTrace = { x: revenueByDate.map(row => row.order_purchase_date), y: revenueByDate.map(row => row.sum_revenue), type: 'scatter', mode: 'lines', line: { color: '#2563eb' } }; const stateTrace = { x: stateRevenue.slice(0, 12).map(row => row.customer_state), y: stateRevenue.slice(0, 12).map(row => row.sum_revenue), type: 'bar', marker: { color: '#2563eb' } }; const categoryTrace = { x: categoryRevenue.slice(0, 12).map(row => row.category_most_common), y: categoryRevenue.slice(0, 12).map(row => row.sum_revenue), type: 'bar', marker: { color: '#7c3aed' } }; const delayTrace = { x: delayByState.slice(0, 12).map(row => row.customer_state), y: delayByState.slice(0, 12).map(row => row.late_rate), type: 'bar', marker: { color: '#f97316' } }; const monthlyTrace = { x: monthlyDelay.map(row => row.purchase_month), y: monthlyDelay.map(row => row.order_count ? row.sum_late / row.order_count : 0), type: 'scatter', mode: 'lines+markers', line: { color: '#16a34a' } }; const revenueLayout = Object.assign({}, document.getElementById('revenue_time_chart').layout || {}, {}); const stateLayout = Object.assign({}, document.getElementById('state_revenue_chart').layout || {}, {}); const categoryLayout = Object.assign({}, document.getElementById('category_revenue_chart').layout || {}, {}); const delayLayout = Object.assign({}, document.getElementById('delay_state_chart').layout || {}, {}); const monthlyLayout = Object.assign({}, document.getElementById('purchase_month_chart').layout || {}, {}); Plotly.react('revenue_time_chart', [revenueTrace], revenueLayout); Plotly.react('state_revenue_chart', [stateTrace], stateLayout); Plotly.react('category_revenue_chart', [categoryTrace], categoryLayout); Plotly.react('delay_state_chart', [delayTrace], delayLayout); Plotly.react('purchase_month_chart', [monthlyTrace], monthlyLayout); }\n")
        f.write("function resetFilters() { stateFilter.value = 'all'; categoryFilter.value = 'all'; dateFrom.value = dateFrom.min; dateTo.value = dateTo.max; applyAllFilters(); }\n")
        f.write("applyFiltersButton.addEventListener('click', applyAllFilters);\n")
        f.write("resetFiltersButton.addEventListener('click', resetFilters);\n")
        f.write("document.addEventListener('DOMContentLoaded', applyAllFilters);\n")
        f.write("</script>\n")
        f.write("</body></html>")
    print(f"Painel salvo em: {dashboard_path.resolve()}")


def train_delay_model(order_base: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, dict[str, float]]:
    df = order_base[order_base["order_status"] == "delivered"].copy()
    target = df["is_late"].astype(int)
    features = df[
        [
            "estimated_delivery_days",
            "freight_value",
            "revenue",
            "avg_item_price",
            "total_items",
            "freight_ratio",
            "distance_km",
            "purchase_weekday",
            "purchase_month",
            "payment_installments",
            "payment_type_primary",
            "customer_state",
            "seller_state",
            "seller_late_rate",
            "customer_late_rate",
            "distinct_categories",
        ]
    ].copy()

    categorical_cols = [
        "payment_type_primary",
        "customer_state",
        "seller_state",
    ]
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy_report": classification_report(y_test, y_pred, digits=4),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print("Modelo treinado com previsão de atraso. Métricas:")
    print(metrics["accuracy_report"])
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    return model, X_test, y_test, metrics


def simulate_monitoring(model: Pipeline, order_base: pd.DataFrame, n: int = 80) -> dict[str, object]:
    recent = order_base[order_base["order_status"] == "delivered"].sample(n, random_state=42)
    X_sim = recent[
        [
            "estimated_delivery_days",
            "freight_value",
            "revenue",
            "avg_item_price",
            "total_items",
            "freight_ratio",
            "distance_km",
            "purchase_weekday",
            "purchase_month",
            "payment_installments",
            "payment_type_primary",
            "customer_state",
            "seller_state",
            "seller_late_rate",
            "customer_late_rate",
            "distinct_categories",
        ]
    ].copy()
    predictions = model.predict(X_sim)
    risk = predictions.mean()
    baseline = order_base[order_base["order_status"] == "delivered"]["is_late"].mean()
    if risk > baseline * 1.1:
        recommendation = (
            "Recomenda retreinar em breve: previsão de atrasos está mais de 10% acima do histórico." 
        )
    elif risk > baseline * 1.05:
        recommendation = (
            "Atenção: monitorar de perto, mas o modelo ainda está dentro de uma faixa aceitável." 
        )
    else:
        recommendation = (
            "Modelo está estável. Retreinar quando ocorrer aumento de atraso acima de 10% ou mudança no mix de vendas." 
        )
    print(f"Simulação de novos pedidos: taxa de atraso esperada = {risk:.2%} para {n} pedidos")
    print(f"Baseline histórico de atraso: {baseline:.2%}")
    print(recommendation)
    return {
        "expected_late_rate": risk,
        "baseline_late_rate": baseline,
        "recommendation": recommendation,
    }


def write_report(metrics: dict[str, object], model_metrics: dict[str, float], monitoring: dict[str, object]) -> None:
    text = textwrap.dedent(
        f"""
        Diagnóstico do negócio Olist
        ---------------------------
        Faturamento total: R$ {metrics['total_revenue']:,.2f}
        Categoria mais vendida: {metrics['top_category']['category_most_common']} (R$ {metrics['top_category']['revenue']:,.2f})
        Estado com maior receita: {metrics['top_state']['customer_state']} (R$ {metrics['top_state']['revenue']:,.2f})
        Estado com maiores atrasos: {metrics['delay_state_rank'][0]['customer_state']} (taxa de atraso {metrics['delay_state_rank'][0]['late_rate']:.2%})

        Métricas do modelo de previsão de atraso:
        ROC AUC: {model_metrics['roc_auc']:.4f}

        Monitoramento e recomendação de retraining:
        Baseline de atraso: {monitoring['baseline_late_rate']:.2%}
        Atraso esperado na simulação: {monitoring['expected_late_rate']:.2%}
        Recomendação: {monitoring['recommendation']}
        """
    )
    Path("report.txt").write_text(text, encoding="utf-8")
    print("Relatório salvo em report.txt")


def main() -> None:
    data_dir = find_dataset_directory()
    if data_dir is None:
        raise FileNotFoundError(
            "Dataset não encontrado. Baixe os arquivos do Kaggle e coloque-os em uma pasta chamada 'dataset' ou 'data' no diretório do projeto. "
            "Os arquivos esperados são: " + ", ".join(sorted(EXPECTED_FILES))
        )

    print(f"Carregando dados de: {data_dir}")
    data = load_data(data_dir)
    order_base = prepare_order_data(data)
    metrics = business_diagnostics(order_base)
    build_dashboard(metrics, order_base)
    model, _, _, model_metrics = train_delay_model(order_base)
    monitoring = simulate_monitoring(model, order_base)
    build_dashboard(metrics, order_base, monitoring['recommendation'])
    write_report(metrics, model_metrics, monitoring)


if __name__ == "__main__":
    main()
