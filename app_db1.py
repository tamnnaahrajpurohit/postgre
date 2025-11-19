# app_postgres_olist_v3.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Company Dashboard (Olist) — Postgres v3", layout="wide")

# ---------- DB config ----------
DEFAULT_DB = "postgresql://postgres:1234@db.hzyzqmyabfqagcxdwjti.supabase.co:5432/postgres"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB).strip()

# cache_resource preferred for non-picklable objects (SQLAlchemy Engine)
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.experimental_singleton

@cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, future=True)

engine = get_engine(DB_URL)

# ---------- helpers ----------
@st.cache_data(ttl=600)
def read_table(table_name, schema="public"):
    try:
        q = text(f'SELECT * FROM "{schema}"."{table_name}"')
        return pd.read_sql(q, engine)
    except Exception:
        try:
            return pd.read_sql(f"select * from {table_name}", engine)
        except Exception:
            return pd.DataFrame()

def safe_to_datetime(s, **kwargs):
    try:
        return pd.to_datetime(s, errors="coerce", **kwargs)
    except Exception:
        return pd.to_datetime(s.astype(str), errors="coerce", **kwargs)

def format_number(x):
    """Format numbers with k (thousand), lac (lakh), cr (crore) suffixes"""
    try:
        if pd.isna(x) or x == 0:
            return "0"
        x = float(x)
        abs_x = abs(x)
        sign = "-" if x < 0 else ""
        
        # Crore (1 crore = 10,000,000)
        if abs_x >= 10000000:
            cr = abs_x / 10000000
            if cr % 1 == 0:
                return f"{sign}{int(cr)} cr"
            else:
                return f"{sign}{cr:.1f} cr"
        # Lakh (1 lakh = 100,000)
        elif abs_x >= 100000:
            lac = abs_x / 100000
            if lac % 1 == 0:
                return f"{sign}{int(lac)} lac"
            else:
                return f"{sign}{lac:.1f} lac"
        # Thousand
        elif abs_x >= 1000:
            k = abs_x / 1000
            if k % 1 == 0:
                return f"{sign}{int(k)}k"
            else:
                return f"{sign}{k:.1f}k"
        # Less than 1000
        else:
            if abs_x % 1 == 0:
                return f"{sign}{int(abs_x)}"
            else:
                return f"{sign}{abs_x:.2f}"
    except Exception:
        return str(x)

def money(x):
    """Format currency with k, lac, cr suffixes"""
    try:
        if pd.isna(x):
            return "R$0"
        x = float(x)
        abs_x = abs(x)
        sign = "-" if x < 0 else ""
        
        # Crore (1 crore = 10,000,000)
        if abs_x >= 10000000:
            cr = abs_x / 10000000
            if cr % 1 == 0:
                return f"R${sign}{int(cr)} cr"
            else:
                return f"R${sign}{cr:.1f} cr"
        # Lakh (1 lakh = 100,000)
        elif abs_x >= 100000:
            lac = abs_x / 100000
            if lac % 1 == 0:
                return f"R${sign}{int(lac)} lac"
            else:
                return f"R${sign}{lac:.1f} lac"
        # Thousand
        elif abs_x >= 1000:
            k = abs_x / 1000
            if k % 1 == 0:
                return f"R${sign}{int(k)}k"
            else:
                return f"R${sign}{k:.1f}k"
        # Less than 1000
        else:
            return f"R${x:,.0f}"
    except Exception:
        return f"R${x:,.0f}" if isinstance(x, (int, float)) else str(x)

def safe_plot(df, fig, key=None):
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty) or fig is None:
        st.info("No data to show this chart.")
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)

def make_table_figure(df, max_rows=50):
    if df is None or df.empty:
        return None
    df2 = df.head(max_rows).copy()
    header = list(df2.columns)
    cells = [df2[col].astype(str).tolist() for col in df2.columns]
    fig = go.Figure(data=[go.Table(header=dict(values=header, align="left"),
                                   cells=dict(values=cells, align="left"))])
    fig.update_layout(height=min(500, 60 + 22 * len(df2)))
    return fig

# ---------- load data ----------
@st.cache_data(ttl=600)
def load_all():
    users = read_table("users")
    warehouses = read_table("warehouses")
    products = read_table("products")
    orders = read_table("orders")
    order_items = read_table("order_items")
    reviews = read_table("reviews")
    returns = read_table("returns")  # Add returns table
    mkt = read_table("marketing_spend")
    return users, warehouses, products, orders, order_items, reviews, returns, mkt

users, warehouses, products, orders, order_items, reviews, returns, mkt = load_all()

# Ensure downstream code can safely reference warehouse_id even if DB column is missing
if isinstance(orders, pd.DataFrame) and 'warehouse_id' not in orders.columns:
    orders['warehouse_id'] = pd.NA

# ---------- Build fact table (merge warehouses to get names) ----------
@st.cache_data(ttl=600)
def build_fact_table(orders, order_items, products, warehouses):
    df = order_items.copy() if isinstance(order_items, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return df

    df['qty'] = df.get('qty', pd.Series(1, index=df.index)).fillna(1).astype(float)
    df['unit_price'] = pd.to_numeric(df.get('unit_price', pd.Series(dtype=float)), errors='coerce')

    ords = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    ords['order_date'] = safe_to_datetime(ords.get('order_date', ords.get('order_purchase_timestamp', pd.Series(dtype='datetime64[ns]'))))
    ords['shipping_cost'] = pd.to_numeric(ords.get('shipping_cost', 0), errors='coerce').fillna(0)
    ords['discount_amount'] = pd.to_numeric(ords.get('discount_amount', 0), errors='coerce').fillna(0)
    ords['channel'] = ords.get('channel', 'online')
    ords['warehouse_id'] = ords.get('warehouse_id', pd.NA)

    orders_cols = [c for c in ['order_id','order_date','discount_amount','shipping_cost','channel','warehouse_id','status'] if c in ords.columns]
    df = df.merge(ords[orders_cols], on='order_id', how='left')

    prod = products.copy() if isinstance(products, pd.DataFrame) else pd.DataFrame()
    if prod.empty:
        prod = df[['product_id']].drop_duplicates().assign(unit_price=np.nan, unit_cost=np.nan, sku=df.get('product_id', pd.Series()).astype(str).str[:8], category='unknown')

    prod['unit_price'] = pd.to_numeric(prod.get('unit_price', pd.Series(dtype=float)), errors='coerce')
    prod['unit_cost'] = pd.to_numeric(prod.get('unit_cost', prod.get('unit_price', 0) * 0.7), errors='coerce').fillna(prod.get('unit_price', 0) * 0.7)
    prod['sku'] = prod.get('sku', prod.get('product_id', pd.Series()).astype(str).str[:8])
    # --- after loading/creating prod dataframe and before merging into df ---
    # Normalize / clean category values to avoid inconsistent names
    if 'category' in prod.columns:
        prod['category'] = prod['category'].astype(str).str.strip().replace({'nan':'unknown', 'None':'unknown'})
        prod.loc[prod['category'].str.len() == 0, 'category'] = 'unknown'
    else:
        prod['category'] = 'unknown'

    prod_keep = [c for c in ['product_id','sku','category','unit_cost','unit_price'] if c in prod.columns]
    df = df.merge(prod[prod_keep].rename(columns={'unit_price':'unit_price_prod'}), on='product_id', how='left')

    # after all computations, ensure fact.category exists and is cleaned
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).str.strip().replace({'nan':'unknown', 'None':'unknown'})
        df.loc[df['category'].str.len() == 0, 'category'] = 'unknown'
    else:
        df['category'] = 'unknown'
    
    # Ensure sku exists in fact table - create from product_id if missing
    if 'sku' not in df.columns:
        df['sku'] = df.get('product_id', pd.Series()).astype(str).str[:8]
    else:
        # Fill any missing sku values with product_id
        df['sku'] = df['sku'].fillna(df.get('product_id', pd.Series()).astype(str).str[:8])

    df['unit_price'] = df['unit_price'].fillna(df.get('unit_price_prod', np.nan))
    df['line_total'] = df['qty'] * df['unit_price']

    line_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['discount_share'] = (df['line_total'] / line_sum) * df['discount_amount'].fillna(0)
    df['net_line_revenue'] = df['line_total'] - df['discount_share'].fillna(0)
    ship_sum = df.groupby('order_id')['line_total'].transform('sum').replace(0, np.nan)
    df['ship_share'] = (df['line_total'] / ship_sum) * df['shipping_cost'].fillna(0)

    # --- FIX: compute cogs safely even if 'unit_cost' column missing ---
    if 'unit_cost' in df.columns:
        unit_cost_series = df['unit_cost'].fillna(df.get('unit_price', 0) * 0.7)
    else:
        unit_cost_series = df.get('unit_price', pd.Series(dtype=float)).fillna(0) * 0.7
    df['cogs'] = (df['qty'] * unit_cost_series).fillna(0)
    # --- end FIX ---

    df['gross_profit'] = df['net_line_revenue'] - df['cogs'] - df['ship_share']
    df['margin_pct'] = np.where(df['net_line_revenue']!=0, df['gross_profit']/df['net_line_revenue'], 0)

    wh = warehouses.copy() if isinstance(warehouses, pd.DataFrame) else pd.DataFrame()
    if not wh.empty:
        keep = [c for c in ['warehouse_id','name','city','state'] if c in wh.columns]
        wh_subset = wh[keep].rename(columns={'name':'warehouse_name','city':'warehouse_city','state':'warehouse_state'})
        df = df.merge(wh_subset, on='warehouse_id', how='left')
    else:
        df['warehouse_name'] = df['warehouse_id']
        df['warehouse_city'] = np.nan
        df['warehouse_state'] = np.nan

    if 'order_date' not in df.columns:
        df['order_date'] = ords.get('order_date', pd.NaT)

    return df

fact = build_fact_table(orders, order_items, products, warehouses)

# ---------- Sidebar filters (extended) ----------
st.sidebar.header("Filters — Advanced")
if isinstance(orders, pd.DataFrame) and 'order_date' in orders.columns and not orders['order_date'].isna().all():
    try:
        min_date = safe_to_datetime(orders['order_date']).min()
        max_date = safe_to_datetime(orders['order_date']).max()
    except Exception:
        min_date, max_date = pd.to_datetime("2000-01-01"), pd.to_datetime("2100-01-01")
else:
    min_date, max_date = pd.to_datetime("2000-01-01"), pd.to_datetime("2100-01-01")
date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date))

channels = ["All"] + sorted(orders['channel'].dropna().unique().tolist()) if isinstance(orders, pd.DataFrame) and 'channel' in orders.columns else ["All"]
channel_sel = st.sidebar.selectbox("Channel", channels, index=0)

if isinstance(warehouses, pd.DataFrame) and 'name' in warehouses.columns:
    wh_names = ["All"] + sorted(warehouses['name'].dropna().astype(str).unique().tolist())
else:
    wh_names = ["All"] + sorted(orders['warehouse_id'].dropna().astype(str).unique().tolist()) if isinstance(orders, pd.DataFrame) and 'warehouse_id' in orders.columns else ["All"]
warehouse_name_sel = st.sidebar.selectbox("Warehouse (Seller name)", wh_names, index=0)

wh_cities = ["All"] + sorted(warehouses['city'].dropna().astype(str).unique().tolist()) if isinstance(warehouses, pd.DataFrame) and 'city' in warehouses.columns else ["All"]
warehouse_city_sel = st.sidebar.selectbox("Warehouse City", wh_cities, index=0)

wh_states = ["All"] + sorted(warehouses['state'].dropna().astype(str).unique().tolist()) if isinstance(warehouses, pd.DataFrame) and 'state' in warehouses.columns else ["All"]
warehouse_state_sel = st.sidebar.selectbox("Warehouse State", wh_states, index=0)

cats = ["All"] + sorted(products['category'].dropna().unique().tolist()) if isinstance(products, pd.DataFrame) and 'category' in products.columns else ["All"]
cat_sel = st.sidebar.selectbox("Product Category", cats, index=0)

sku_text = st.sidebar.text_input("SKU contains (text, leave empty = all)")
product_id_text = st.sidebar.text_input("Product ID contains (text)")

rating_min, rating_max = st.sidebar.slider("Rating range", 0, 5, (0,5))
min_rev = st.sidebar.number_input("Min revenue per order (R$)", value=0.0, step=100.0)
max_rev = st.sidebar.number_input("Max revenue per order (R$) (0 = no max)", value=0.0, step=100.0)
qty_min = st.sidebar.number_input("Min qty per line", value=0, step=1)
qty_max = st.sidebar.number_input("Max qty per line (0 = no max)", value=0, step=1)

# ---------- Apply filters ----------
f = fact.copy() if isinstance(fact, pd.DataFrame) else pd.DataFrame()
if not f.empty:
    f['order_date'] = safe_to_datetime(f['order_date'])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(f['order_date'] >= s) & (f['order_date'] <= e)]
    if channel_sel != "All" and 'channel' in f.columns:
        f = f[f['channel'] == channel_sel]
    if warehouse_name_sel != "All" and 'warehouse_name' in f.columns:
        f = f[f['warehouse_name'].astype(str) == str(warehouse_name_sel)]
    if warehouse_city_sel != "All" and 'warehouse_city' in f.columns:
        f = f[f['warehouse_city'].astype(str) == str(warehouse_city_sel)]
    if warehouse_state_sel != "All" and 'warehouse_state' in f.columns:
        f = f[f['warehouse_state'].astype(str) == str(warehouse_state_sel)]
    if cat_sel != "All" and 'category' in f.columns:
        f = f[f['category'] == cat_sel]
    if sku_text:
        f = f[f['sku'].astype(str).str.contains(sku_text, case=False, na=False)]
    if product_id_text:
        f = f[f['product_id'].astype(str).str.contains(product_id_text, case=False, na=False)]
    if qty_min > 0:
        f = f[f['qty'] >= qty_min]
    if qty_max > 0:
        f = f[f['qty'] <= qty_max]
    if min_rev > 0 or max_rev > 0:
        ord_rev = f.groupby("order_id")["net_line_revenue"].sum().rename("rev_per_order").reset_index()
        if min_rev > 0:
            keep_ids = ord_rev[ord_rev['rev_per_order'] >= min_rev]['order_id']
            f = f[f['order_id'].isin(keep_ids)]
        if max_rev > 0:
            keep_ids = ord_rev[ord_rev['rev_per_order'] <= max_rev]['order_id']
            f = f[f['order_id'].isin(keep_ids)]
    # Rating filter - merge with reviews if available
    if (rating_min > 0 or rating_max < 5) and isinstance(reviews, pd.DataFrame) and not reviews.empty and 'order_id' in reviews.columns and 'rating' in reviews.columns:
        rev_filtered = reviews[(reviews['rating'] >= rating_min) & (reviews['rating'] <= rating_max)]
        if not rev_filtered.empty:
            order_ids_with_rating = rev_filtered['order_id'].unique()
            f = f[f['order_id'].isin(order_ids_with_rating)]

# ---------- Dashboard header ----------
st.title("Company Dashboard — Insights & Profitability")

if f.empty:
    st.warning("No data in fact after applying filters — check tables and filter selections.")
else:
    total_rev = f["net_line_revenue"].sum()
    gross_profit = f["gross_profit"].sum()
    orders_count = f["order_id"].nunique()
    aov = f.groupby("order_id")["net_line_revenue"].sum().mean()
    margin_pct = (gross_profit / total_rev) if total_rev != 0 else 0

    rr = 0.0
    if isinstance(orders, pd.DataFrame) and "status" in orders.columns:
        o2 = orders.copy()
        o2['is_returned'] = o2['status'].isin(["canceled", "unavailable"]).astype(int)
        tot = o2['order_id'].nunique()
        ret = o2[o2['is_returned'] == 1]['order_id'].nunique()
        rr = (ret / tot) if tot > 0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue (filtered)", money(total_rev))
    k2.metric("Gross Profit", money(gross_profit))
    k3.metric("Gross Margin %", f"{margin_pct*100:.2f}%")
    k4.metric("Orders", format_number(orders_count))
    k5.metric("AOV", money(aov) if not np.isnan(aov) else "N/A")
    k6.metric("Return Rate", f"{rr*100:.2f}%")

st.markdown("---")

tab_overview, tab_products, tab_customers, tab_ops, tab_marketing, tab_reviews = st.tabs(["Overview", "Products & Categories", "Customers", "Operations", "Marketing", "Reviews"])

with tab_overview:
    st.header("Overview — Revenue & Profitability Trends")
    if not f.empty:
        f["order_date"] = pd.to_datetime(f["order_date"], errors="coerce")
        f["week"] = f["order_date"].dt.to_period("W").astype(str)
        f["month"] = f["order_date"].dt.to_period("M").astype(str)

        rev_ts = f.groupby(pd.Grouper(key="order_date", freq="W"))["net_line_revenue"].sum().reset_index()
        fig = px.line(rev_ts, x="order_date", y="net_line_revenue", markers=True, title="Revenue (Weekly)")
        fig.update_layout(yaxis_tickprefix="R$")
        st.plotly_chart(fig, use_container_width=True, key="plot_01")

        rev_m = f.groupby("month")["net_line_revenue"].sum().rename("revenue").reset_index()
        gp_m = f.groupby("month")["gross_profit"].sum().rename("gross_profit").reset_index()
        both = rev_m.merge(gp_m, on="month", how="left").fillna(0)
        fig2 = px.bar(both, x="month", y=["revenue","gross_profit"], barmode="group", title="Revenue vs Gross Profit (Monthly)")
        fig2.update_layout(yaxis_tickprefix="R$")
        st.plotly_chart(fig2, use_container_width=True, key="plot_02")

        col1, col2 = st.columns(2)
        with col1:
            rev_per_order = f.groupby("order_id")["net_line_revenue"].sum().reset_index()
            fig3 = px.histogram(rev_per_order, x="net_line_revenue", nbins=60, title="Revenue per Order Distribution")
            fig3.update_layout(yaxis_title="Count", xaxis_title="Revenue per Order", xaxis_tickprefix="R$")
            st.plotly_chart(fig3, use_container_width=True, key="plot_03")
        with col2:
            aov_by_month = f.groupby("month").apply(lambda x: x.groupby("order_id")["net_line_revenue"].sum().mean()).reset_index(name="AOV")
            fig4 = px.line(aov_by_month, x="month", y="AOV", markers=True, title="AOV (Monthly)")
            st.plotly_chart(fig4, use_container_width=True, key="plot_04")

        kp = both.tail(12).melt(id_vars="month", value_vars=["revenue","gross_profit"], var_name="metric", value_name="value")
        fig_kp = px.area(kp, x="month", y="value", color="metric", facet_col="metric", title="Last 12 Months: Revenue & Gross Profit (area)")
        st.plotly_chart(fig_kp, use_container_width=True, key="plot_05")
    else:
        st.info("No transactional data available for overview.")

with tab_products:
    st.header("Products & Category Profitability")
    if not f.empty:
        prod_sum = f.groupby(["product_id","sku","category"]).agg(
            revenue=("net_line_revenue","sum"),
            margin=("gross_profit","sum"),
            qty=("qty","sum"),
            margin_pct=("margin_pct","mean")
        ).reset_index().sort_values("revenue", ascending=False)

        st.subheader("Top 15 Products by Revenue")
        top15 = prod_sum.head(15)
        fig_p = px.bar(top15, x="sku", y="revenue", hover_data=["margin","qty"], title="Top 15 Products (Revenue)")
        fig_p.update_layout(yaxis_tickprefix="R$")
        st.plotly_chart(fig_p, use_container_width=True, key="plot_06")
        st.dataframe(top15.style.format({"revenue":"{:.0f}","margin":"{:.0f}","margin_pct":"{:.2%}"}), height=300)

        st.subheader("Category Profitability (Revenue vs Margin)")
        cat_sc = f.groupby("category").agg(revenue=("net_line_revenue","sum"), margin=("gross_profit","sum")).reset_index()
        fig_cat = px.scatter(cat_sc, x="revenue", y="margin", size="revenue", hover_data=["category"], title="Category: Revenue vs Margin")
        fig_cat.update_layout(yaxis_tickprefix="R$", xaxis_tickprefix="R$")
        st.plotly_chart(fig_cat, use_container_width=True, key="plot_07")

        st.subheader("Products Losing Money (Negative Gross Profit)")
        loss_prods = prod_sum[prod_sum["margin"]<0].sort_values("margin").head(20)
        if not loss_prods.empty:
            st.plotly_chart(px.bar(loss_prods, x="sku", y="margin", title="Top Loss-Making SKUs"), use_container_width=True, key="plot_08")
            st.dataframe(loss_prods.style.format({"revenue":"{:.0f}","margin":"{:.0f}"}))
        else:
            st.info("No negative-margin products in filtered data.")

        st.subheader("Unit Price vs Unit Cost (Products)")
        p2 = products.copy() if isinstance(products, pd.DataFrame) and not products.empty else pd.DataFrame()
        if not p2.empty and "unit_price" in p2.columns and "unit_cost" in p2.columns:
            st.plotly_chart(px.scatter(p2, x="unit_price", y="unit_cost", hover_data=["sku","category"], title="Unit Price vs Unit Cost"), use_container_width=True, key="plot_09")
        else:
            st.info("Product price/cost data missing to show Price vs Cost chart.")
    else:
        st.info("No product transactions to analyze.")

with tab_customers:
    st.header("Customer Insights & Retention")
    if isinstance(users, pd.DataFrame) and not users.empty and not f.empty:
        u = users.copy()
        if 'created_at' in u.columns:
            u["cohort_month"] = pd.to_datetime(u["created_at"], errors="coerce").dt.to_period("M").astype(str)
            cohort = u.groupby("cohort_month")["user_id"].nunique().reset_index(name="new_users")
            st.subheader("New Users by Month")
            st.plotly_chart(px.bar(cohort, x="cohort_month", y="new_users", title="New users per month"), use_container_width=True, key="plot_10")
        else:
            st.info("created_at column missing in users to compute cohorts.")

        orders_by_user = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
        if not orders_by_user.empty and "customer_id" in orders_by_user.columns:
            ob = orders_by_user.groupby("customer_id")["order_id"].nunique().reset_index(name="orders_per_user")
            repeat_rate = (ob["orders_per_user"]>1).mean()
            st.metric("Repeat Purchase Rate", f"{repeat_rate*100:.2f}%")
            st.subheader("Top Cities by Users")
            if "city" in users.columns:
                uc = users["city"].value_counts().head(20).reset_index()
                uc.columns = ["city","users"]
                st.plotly_chart(px.bar(uc, x="city", y="users", title="Top Cities by Registered Users"), use_container_width=True, key="plot_11")
            else:
                st.info("City column missing in users data.")
        else:
            st.info("Customer-level order data missing to compute repeat purchase rate.")
    else:
        st.info("User or transaction data missing for customer insights.")

with tab_ops:
    st.header("Operations — Returns, Shipping & Discounts")
    orders2 = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    if "order_date" not in orders2.columns and "order_purchase_timestamp" in orders2.columns:
        orders2["order_date"] = pd.to_datetime(orders2["order_purchase_timestamp"], errors="coerce")
    if not f.empty:
        if "status" in orders2.columns:
            orders2["month"] = pd.to_datetime(orders2["order_date"], errors="coerce").dt.to_period("M").astype(str)
            tot = orders2.groupby("month")["order_id"].nunique().rename("orders_total")
            ret = orders2[orders2["status"].isin(["canceled","unavailable"])].groupby("month")["order_id"].nunique().rename("orders_returned")
            rr_df = pd.concat([tot, ret], axis=1).fillna(0).reset_index()
            rr_df["return_rate"] = (rr_df["orders_returned"] / rr_df["orders_total"]).replace([np.inf, np.nan], 0)
            st.subheader("Return / Cancel Rate by Month")
            st.plotly_chart(px.line(rr_df, x="month", y="return_rate", markers=True, title="Return Rate by Month"), use_container_width=True, key="plot_12")

        st.subheader("Shipping Cost vs Order Revenue")
        ord_rev = f.groupby("order_id")[["net_line_revenue","ship_share","discount_share"]].sum().reset_index()
        ord_rev = ord_rev.rename(columns={"ship_share":"shipping_cost","discount_share":"discount_amount"})
        fig_ship = px.scatter(ord_rev, x="net_line_revenue", y="shipping_cost", title="Shipping vs Order Revenue", hover_data=["discount_amount"])
        fig_ship.update_layout(yaxis_tickprefix="R$", xaxis_tickprefix="R$")
        st.plotly_chart(fig_ship, use_container_width=True, key="plot_13")

        st.subheader("Discount Impact: Discount vs Revenue per Order")
        fig_disc = px.scatter(ord_rev, x="net_line_revenue", y="discount_amount", title="Discount vs Order Revenue")
        st.plotly_chart(fig_disc, use_container_width=True, key="plot_14")
    else:
        st.info("No transactional data for operations metrics.")

with tab_marketing:
    st.header("Marketing — Spend & Synthetic CPA")
    if isinstance(mkt, pd.DataFrame) and not mkt.empty and not f.empty:
        f["month"] = f["order_date"].dt.to_period("M").astype(str)
        ord_by_chan = f.groupby(["month","channel"])["order_id"].nunique().reset_index(name="orders")
        mk = mkt.merge(ord_by_chan, on=["month","channel"], how="left").fillna({"orders":0})
        mk["cpa"] = mk["spend"] / mk["orders"].replace(0, np.nan)
        st.subheader("Marketing Spend by Channel (Monthly)")
        st.plotly_chart(px.bar(mk, x="month", y="spend", color="channel", barmode="group", title="Marketing Spend"), use_container_width=True, key="plot_15")
        st.subheader("CPA (Cost per Order) by Channel")
        st.plotly_chart(px.bar(mk, x="month", y="cpa", color="channel", barmode="group", title="CPA by Channel"), use_container_width=True, key="plot_16")
        st.dataframe(mk[["month","channel","spend","orders","cpa"]].sort_values(["month","channel"]), height=300)
    else:
        st.info("No marketing spend file or not enough data to compute CPA.")

with tab_reviews:
    st.header("Reviews — Customer Feedback & Ratings")
    
    def nonempty_reviews(df):
        return (df is not None) and isinstance(df, pd.DataFrame) and (len(df) > 0)
    
    if nonempty_reviews(reviews):
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_reviews = len(reviews)
            st.metric("Total Reviews", format_number(total_reviews))
        with col2:
            if "rating" in reviews.columns:
                avg_rating = reviews["rating"].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}")
            else:
                st.metric("Average Rating", "N/A")
        with col3:
            if "review_date" in reviews.columns:
                reviews_with_date = reviews["review_date"].notna().sum()
                st.metric("Reviews with Date", format_number(reviews_with_date))
            else:
                st.metric("Reviews with Date", "N/A")
        with col4:
            if "product_id" in reviews.columns:
                unique_products = reviews["product_id"].nunique()
                st.metric("Products Reviewed", format_number(unique_products))
            else:
                st.metric("Products Reviewed", "N/A")
        
        st.markdown("---")
        
        # Rating Distribution
        st.subheader("01 — Rating Distribution")
        if "rating" in reviews.columns:
            rd = reviews["rating"].value_counts().sort_index().reset_index()
            rd.columns = ["rating", "count"]
            st.plotly_chart(px.bar(rd, x="rating", y="count", title="Rating Distribution", 
                                  labels={"rating": "Rating", "count": "Number of Reviews"}), 
                          use_container_width=True, key="rev_plot_01")
            
            # Rating histogram
            st.subheader("02 — Rating Histogram")
            st.plotly_chart(px.histogram(reviews, x="rating", nbins=6, title="Rating Histogram",
                                        labels={"rating": "Rating", "count": "Frequency"}), 
                          use_container_width=True, key="rev_plot_02")
        else:
            st.info("No rating column in reviews data.")
        
        # Reviews per Month
        st.subheader("03 — Reviews per Month")
        if "review_date" in reviews.columns:
            r2 = reviews.copy()
            r2["review_date"] = pd.to_datetime(r2["review_date"], errors="coerce")
            r2["month"] = r2["review_date"].dt.to_period("M").astype(str)
            rpm = r2.groupby("month")["review_id"].nunique().reset_index(name="reviews")
            st.plotly_chart(px.line(rpm, x="month", y="reviews", markers=True, 
                                   title="Reviews per Month",
                                   labels={"month": "Month", "reviews": "Number of Reviews"}), 
                          use_container_width=True, key="rev_plot_03")
        else:
            st.info("No review_date column in reviews data.")
        
        # Return Rate vs Rating
        st.subheader("04 — Return Rate vs Rating")
        orders2 = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
        if "status" in orders2.columns and "order_id" in reviews.columns and "rating" in reviews.columns:
            orders2["order_date"] = pd.to_datetime(orders2.get("order_date", orders2.get("order_purchase_timestamp", pd.Series(dtype='datetime64[ns]'))), errors="coerce")
            orr = orders2[["order_id", "status"]].copy()
            orr["is_returned"] = orr["status"].isin(["canceled", "unavailable", "returned"]).astype(int)
            rj = reviews.merge(orr, on="order_id", how="left")
            byrat = rj.groupby("rating")["is_returned"].mean().reset_index()
            byrat.columns = ["rating", "return_rate"]
            st.plotly_chart(px.line(byrat, x="rating", y="return_rate", markers=True, 
                                   title="Return Rate vs Rating",
                                   labels={"rating": "Rating", "return_rate": "Return Rate"}), 
                          use_container_width=True, key="rev_plot_04")
        else:
            st.info("Cannot compute Return Rate vs Rating (missing reviews, order_id, rating, or order status).")
        
        # Top Reviewed Products
        st.subheader("05 — Top Reviewed Products")
        if "product_id" in reviews.columns:
            top_reviewed = reviews["product_id"].value_counts().head(20).reset_index()
            top_reviewed.columns = ["product_id", "review_count"]
            st.plotly_chart(px.bar(top_reviewed, x="product_id", y="review_count", 
                                   title="Top 20 Most Reviewed Products",
                                   labels={"product_id": "Product ID", "review_count": "Number of Reviews"}), 
                          use_container_width=True, key="rev_plot_05")
        else:
            st.info("No product_id column in reviews data.")
        
        # Average Rating by Product
        st.subheader("06 — Average Rating by Product (Top 20)")
        if "product_id" in reviews.columns and "rating" in reviews.columns:
            avg_rating_prod = reviews.groupby("product_id")["rating"].agg(["mean", "count"]).reset_index()
            avg_rating_prod.columns = ["product_id", "avg_rating", "review_count"]
            avg_rating_prod = avg_rating_prod[avg_rating_prod["review_count"] >= 2].sort_values("avg_rating", ascending=False).head(20)
            st.plotly_chart(px.bar(avg_rating_prod, x="product_id", y="avg_rating", 
                                   title="Average Rating by Product (min 2 reviews)",
                                   labels={"product_id": "Product ID", "avg_rating": "Average Rating"}), 
                          use_container_width=True, key="rev_plot_06")
        else:
            st.info("Cannot compute average rating by product (missing product_id or rating).")
        
        # Order Revenue by Rating
        st.subheader("07 — Order Revenue by Rating")
        if "order_id" in reviews.columns and "rating" in reviews.columns and not f.empty:
            order_rev = f.groupby("order_id")["net_line_revenue"].sum().reset_index()
            order_rev.columns = ["order_id", "order_revenue"]
            rev_rating = reviews.merge(order_rev, on="order_id", how="left")
            if not rev_rating.empty and "order_revenue" in rev_rating.columns:
                st.plotly_chart(px.scatter(rev_rating, x="rating", y="order_revenue", 
                                         title="Order Revenue by Rating",
                                         labels={"rating": "Rating", "order_revenue": "Order Revenue (R$)"}), 
                              use_container_width=True, key="rev_plot_07")
            else:
                st.info("Cannot merge reviews with order revenue data.")
        else:
            st.info("Cannot compute Order Revenue by Rating (missing order_id, rating, or fact table).")
        
        # Rating vs Product Category
        st.subheader("08 — Average Rating by Category")
        if "product_id" in reviews.columns and "rating" in reviews.columns and not products.empty and "category" in products.columns:
            rev_prod = reviews.merge(products[["product_id", "category"]], on="product_id", how="left")
            cat_rating = rev_prod.groupby("category")["rating"].mean().sort_values(ascending=False).reset_index()
            cat_rating.columns = ["category", "avg_rating"]
            st.plotly_chart(px.bar(cat_rating, x="category", y="avg_rating", 
                                   title="Average Rating by Category",
                                   labels={"category": "Category", "avg_rating": "Average Rating"}), 
                          use_container_width=True, key="rev_plot_08")
        else:
            st.info("Cannot compute average rating by category (missing required columns).")
        
        # Reviews over Time (if review_date exists)
        if "review_date" in reviews.columns:
            st.subheader("09 — Reviews Trend Over Time")
            r3 = reviews.copy()
            r3["review_date"] = pd.to_datetime(r3["review_date"], errors="coerce")
            r3 = r3[r3["review_date"].notna()]
            if not r3.empty:
                r3["date"] = r3["review_date"].dt.date
                daily_reviews = r3.groupby("date")["review_id"].nunique().reset_index()
                daily_reviews.columns = ["date", "reviews"]
                st.plotly_chart(px.line(daily_reviews, x="date", y="reviews", markers=True,
                                       title="Daily Reviews Trend",
                                       labels={"date": "Date", "reviews": "Number of Reviews"}), 
                              use_container_width=True, key="rev_plot_09")
        
        # Review Text Sample (if available)
        if "review_text" in reviews.columns:
            st.subheader("10 — Sample Reviews")
            sample_reviews = reviews[["review_id", "order_id", "product_id", "rating", "review_text"]].head(100)
            st.dataframe(sample_reviews, height=300, use_container_width=True)
        
    else:
        st.info("No reviews data available from PostgreSQL database.")

with st.expander("Show all 25 visuals (full):"):
    def render_25_visuals(users, warehouses, products, orders, order_items, reviews, returns, fact):
        st.header("All 25 Visuals (Company-wide)")

        def nonempty(df):
            return (df is not None) and (len(df) > 0)

        f = fact.copy() if isinstance(fact, pd.DataFrame) else pd.DataFrame()
        if f.empty:
            st.info("No data to render the visuals.")
            return

        f["order_date"] = pd.to_datetime(f["order_date"], errors="coerce")
        f["month"] = f["order_date"].dt.to_period("M").astype(str)
        orders2 = orders.copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
        if "order_date" not in orders2.columns and "order_purchase_timestamp" in orders2.columns:
            orders2["order_date"] = pd.to_datetime(orders2["order_purchase_timestamp"], errors="coerce")
        orders2["month"] = pd.to_datetime(orders2["order_date"], errors="coerce").dt.to_period("M").astype(str)

        st.subheader("01 — Revenue per Order (Histogram)")
        rev_per_order = f.groupby("order_id")["net_line_revenue"].sum().reset_index()
        st.plotly_chart(px.histogram(rev_per_order, x="net_line_revenue", nbins=50, title="Revenue per Order"), use_container_width=True, key="plot_17")

        st.subheader("02 — Orders by Status")
        if "status" in orders2.columns:
            s = orders2["status"].value_counts().reset_index()
            s.columns = ["status","count"]
            st.plotly_chart(px.bar(s, x="status", y="count", title="Orders by Status"), use_container_width=True, key="plot_18")
        else:
            st.info("No status column found in orders.")

        st.subheader("03 — Average Order Value (Monthly)")
        aov_m = f.groupby("month")["net_line_revenue"].sum().reset_index()
        ord_m = f.groupby("month")["order_id"].nunique().reset_index()
        aov_m = aov_m.merge(ord_m, on="month", how="left")
        aov_m["AOV"] = aov_m["net_line_revenue"] / aov_m["order_id"]
        st.plotly_chart(px.line(aov_m, x="month", y="AOV", markers=True, title="AOV Monthly"), use_container_width=True, key="plot_19")

        st.subheader("04 — Revenue by Category (Top 15)")
        cat_rev = f.groupby("category")["net_line_revenue"].sum().sort_values(ascending=False).head(15).reset_index()
        st.plotly_chart(px.bar(cat_rev, x="category", y="net_line_revenue", title="Revenue by Category").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_20")

        st.subheader("05 — Gross Profit by Category")
        cat_gp = f.groupby("category")["gross_profit"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(cat_gp, x="category", y="gross_profit", title="Gross Profit by Category").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_21")

        st.subheader("06 — Revenue by Warehouse (Seller)")
        wh_rev = f.groupby("warehouse_id")["net_line_revenue"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(wh_rev, x="warehouse_id", y="net_line_revenue", title="Revenue by Warehouse").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_22")

        st.subheader("07 — Gross Profit by Warehouse")
        wh_gp = f.groupby("warehouse_id")["gross_profit"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(wh_gp, x="warehouse_id", y="gross_profit", title="Gross Profit by Warehouse").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_23")

        st.subheader("08 — Revenue by Channel")
        if "channel" in f.columns:
            ch = f.groupby("channel")["net_line_revenue"].sum().reset_index()
            st.plotly_chart(px.bar(ch, x="channel", y="net_line_revenue", title="Revenue by Channel").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_24")

        st.subheader("09 — Weekly Revenue Trend")
        weekly = f.groupby(pd.Grouper(key="order_date", freq="W"))["net_line_revenue"].sum().reset_index()
        st.plotly_chart(px.line(weekly, x="order_date", y="net_line_revenue", markers=True, title="Weekly Revenue"), use_container_width=True, key="plot_25")

        st.subheader("10 — Monthly Gross Profit Trend")
        gp_m = f.groupby("month")["gross_profit"].sum().reset_index()
        st.plotly_chart(px.line(gp_m, x="month", y="gross_profit", markers=True, title="Monthly Gross Profit").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_26")

        st.subheader("11 — Top Products by Revenue (Top 15)")
        top_prod_rev = f.groupby(["product_id","sku"]).agg(revenue=("net_line_revenue","sum")).reset_index().sort_values("revenue", ascending=False).head(15)
        st.plotly_chart(px.bar(top_prod_rev, x="sku", y="revenue", title="Top Products by Revenue").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_27")

        st.subheader("12 — Top Products by Margin (Top 15)")
        top_prod_mg = f.groupby(["product_id","sku"]).agg(margin=("gross_profit","sum")).reset_index().sort_values("margin", ascending=False).head(15)
        st.plotly_chart(px.bar(top_prod_mg, x="sku", y="margin", title="Top Products by Margin").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_28")

        st.subheader("13 — Popularity vs Profitability")
        pq = f.groupby(["product_id","sku"]).agg(qty=("qty","sum"), avg_margin=("gross_profit","mean")).reset_index()
        st.plotly_chart(px.scatter(pq, x="qty", y="avg_margin", hover_data=["sku"], title="Qty vs Avg Margin"), use_container_width=True, key="plot_29")

        st.subheader("14 — Price vs Unit Cost (Products)")
        p2 = products.copy() if isinstance(products, pd.DataFrame) and not products.empty else pd.DataFrame()
        if not p2.empty and "unit_price" in p2.columns and "unit_cost" in p2.columns:
            st.plotly_chart(px.scatter(p2, x="unit_price", y="unit_cost", hover_data=["sku","category"], title="Unit Price vs Unit Cost"), use_container_width=True, key="plot_30")
        else:
            st.info("Unit price/cost missing for Price vs Cost chart.")

        st.subheader("15 — Category Revenue Share")
        st.plotly_chart(px.pie(cat_rev, names="category", values="net_line_revenue", title="Category Revenue Share"), use_container_width=True, key="plot_31")

        st.subheader("16 — Return/Cancel Rate by Month")
        o = orders2.copy() if isinstance(orders2, pd.DataFrame) else pd.DataFrame()
        if "status" in o.columns:
            tot = o.groupby("month")["order_id"].nunique().rename("orders_total")
            ret = o[o["status"].isin(["canceled","unavailable"])].groupby("month")["order_id"].nunique().rename("orders_returned")
            rr = pd.concat([tot, ret], axis=1).fillna(0).reset_index()
            rr["return_rate"] = (rr["orders_returned"] / rr["orders_total"]).replace([np.inf, np.nan], 0)
            st.plotly_chart(px.line(rr, x="month", y="return_rate", markers=True, title="Return Rate by Month"), use_container_width=True, key="plot_32")
        else:
            st.info("No status column in orders to compute return rates.")

        st.subheader("17 — Rating Distribution")
        if nonempty(reviews) and "rating" in reviews.columns:
            rd = reviews["rating"].value_counts().sort_index().reset_index()
            rd.columns = ["rating","count"]
            st.plotly_chart(px.bar(rd, x="rating", y="count", title="Rating Distribution"), use_container_width=True, key="plot_33")
        else:
            st.info("No reviews data to show rating distribution.")

        st.subheader("18 — Reviews per Month")
        if nonempty(reviews) and "review_date" in reviews.columns:
            r2 = reviews.copy()
            r2["rm"] = pd.to_datetime(r2["review_date"], errors="coerce").dt.to_period("M").astype(str)
            rpm = r2.groupby("rm")["review_id"].nunique().reset_index(name="reviews")
            st.plotly_chart(px.line(rpm, x="rm", y="reviews", markers=True, title="Reviews per Month"), use_container_width=True, key="plot_34")
        else:
            st.info("No reviews date data available.")

        st.subheader("19 — Return Rate vs Rating")
        if nonempty(reviews) and "status" in orders2.columns:
            orr = orders2[["order_id","status"]].copy()
            orr["is_returned"] = orr["status"].isin(["canceled","unavailable"]).astype(int)
            rj = reviews.merge(orr, on="order_id", how="left")
            byrat = rj.groupby("rating")["is_returned"].mean().reset_index()
            st.plotly_chart(px.line(byrat, x="rating", y="is_returned", markers=True, title="Return Rate vs Rating"), use_container_width=True, key="plot_35")
        else:
            st.info("Cannot compute Return Rate vs Rating (missing reviews or order status).")

        st.subheader("20 — Top Cities by Users")
        if nonempty(users) and "city" in users.columns:
            uc = users["city"].value_counts().head(20).reset_index()
            uc.columns = ["city","users"]
            st.plotly_chart(px.bar(uc, x="city", y="users", title="Top Cities by Users"), use_container_width=True, key="plot_36")
        else:
            st.info("Users data missing or no city column.")

        st.subheader("21 — Orders by Customer State")
        if isinstance(orders, pd.DataFrame) and "customer_state" in orders.columns:
            st_state = orders["customer_state"].value_counts().reset_index()
            st_state.columns = ["state","orders"]
            st.plotly_chart(px.bar(st_state, x="state", y="orders", title="Orders by Customer State"), use_container_width=True, key="plot_37")
        else:
            st.info("No customer_state column in orders.")

        st.subheader("22 — Shipping Cost vs Order Revenue")
        ord_rev = f.groupby("order_id")[["net_line_revenue","ship_share"]].sum().reset_index()
        ord_rev["shipping_cost"] = ord_rev["ship_share"]
        st.plotly_chart(px.scatter(ord_rev, x="net_line_revenue", y="shipping_cost", title="Shipping Cost vs Order Revenue"), use_container_width=True, key="plot_38")

        st.subheader("23 — Discount vs Order Revenue")
        disc_ord = f.groupby("order_id")[["net_line_revenue","discount_share"]].sum().reset_index()
        disc_ord["discount_amount"] = disc_ord["discount_share"]
        st.plotly_chart(px.scatter(disc_ord, x="net_line_revenue", y="discount_amount", title="Discount vs Order Revenue"), use_container_width=True, key="plot_39")

        st.subheader("24 — Revenue vs Gross Profit (Monthly)")
        rev_m = f.groupby("month")["net_line_revenue"].sum().rename("revenue").reset_index()
        mg_m = f.groupby("month")["gross_profit"].sum().rename("gross_profit").reset_index()
        both = rev_m.merge(mg_m, on="month", how="left")
        st.plotly_chart(px.bar(both, x="month", y=["revenue","gross_profit"], barmode="group", title="Revenue & Gross Profit (Monthly)").update_layout(yaxis_tickprefix="R$"), use_container_width=True, key="plot_40")

        st.subheader("25 — Category-level Revenue vs Margin")
        cat_sc = f.groupby("category").agg(revenue=("net_line_revenue","sum"), margin=("gross_profit","sum")).reset_index()
        st.plotly_chart(px.scatter(cat_sc, x="revenue", y="margin", hover_data=["category"], title="Category Revenue vs Margin").update_layout(xaxis_tickprefix="R$", yaxis_tickprefix="R$"), use_container_width=True, key="plot_41")

    render_25_visuals(users, warehouses, products, orders, order_items, reviews, returns, fact)

st.markdown("---")
st.caption("Tip: Use the sidebar filters (date, channel, warehouse, category) to drill down. Charts update with filters.")

