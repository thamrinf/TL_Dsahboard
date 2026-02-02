import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
import geopandas as gpd
import folium
from streamlit_folium import st_folium


# 1. GLOBAL CONFIG
st.set_page_config(page_title="Activity Portfolio Dashboard", layout="wide")


# 2. CACHED DATA LOADERS
@st.cache_data
def load_data():
    df = pd.read_csv("data/activities.csv")
    # Generate unique ID for records
    df["activity_group"] = (
        df["Municipality"].astype(str) + " | " + 
        df["Sector"].astype(str) + " | " + 
        df["Implementing_Partner"].fillna("Unknown") + " | " + 
        df["Funding_Donor"].fillna("Unknown") + " | " + 
        df["YearStart"].astype(str) + " | " + 
        df["YearEnd"].fillna(0).astype(int).astype(str) + " | " + 
        df["Status"] + " | " + 
        df["Budget"].round(2).astype(str)
    )
    df["activity_id"] = pd.util.hash_pandas_object(df["activity_group"], index=False)
    return df

@st.cache_data
def load_geo():
    gdf = gpd.read_file("data/tls_admin1.geojson")
    gdf = gdf.to_crs(epsg=4326)
    gdf["mun_clean"] = gdf["adm1_name"].str.strip().str.upper()
    return gdf

# 1. LOAD DATA
df_raw = load_data()
gdf_raw = load_geo()

# (Sidebar filters from previous step should use df_raw)

# 2. DEFINE SIDEBAR FILTERS (Must come BEFORE filtering logic)
st.sidebar.header("Global Filters")

year_range = st.sidebar.slider(
    "Start Year Range",
    int(df_raw["YearStart"].min()),
    int(df_raw["YearStart"].max()),
    (int(df_raw["YearStart"].min()), int(df_raw["YearStart"].max()))
)

sector_filter = st.sidebar.multiselect("Sector", sorted(df_raw["Sector"].dropna().unique()))
municipality_filter = st.sidebar.multiselect("Municipality", sorted(df_raw["Municipality"].dropna().unique()))
donor_filter = st.sidebar.multiselect("Donor", sorted(df_raw["Funding_Donor"].dropna().unique()))

# 3. CREATE FILTERED DATAFRAME
filtered_df = df_raw[
    (df_raw["YearStart"] >= year_range[0]) &
    (df_raw["YearStart"] <= year_range[1])
].copy()

if sector_filter:
    filtered_df = filtered_df[filtered_df["Sector"].isin(sector_filter)]
if municipality_filter:
    filtered_df = filtered_df[filtered_df["Municipality"].isin(municipality_filter)]
if donor_filter:
    filtered_df = filtered_df[filtered_df["Funding_Donor"].isin(donor_filter)]

# 4. SIDEBAR STATS (Consolidated into one block)
st.sidebar.divider()
st.sidebar.metric("Filtered Activities", f"{len(filtered_df):,}")

if not df_raw.empty:
    coverage = (len(filtered_df) / len(df_raw)) * 100
    st.sidebar.progress(coverage / 100) # Added a nice visual bar
    st.sidebar.caption(f"Showing {coverage:.1f}% of total portfolio")
else:
    st.sidebar.caption("No data available.")


# --- APP HEADER ---
st.title("Development Activity Portfolio Dashboard")

tabs = st.tabs([
    "Methodology", "Data Overview", "Portfolio Overview", 
    "Geographic Footprint", "Sector Allocation", "Donors & Implementers", 
    "Time Dynamics", "Map", "SDGs Alignment", "Municipality Analysis"
])

# ==========================================
# TAB 0: METHODOLOGY
# ==========================================

with tabs[0]:
    st.subheader("Methodology & Data Notes")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("##### Unit of Analysis")
        st.write(
            "Each row represents an **activity record**. Records represent individual activity instances. "
            "Similar attributes do not necessarily indicate a single project, as no unique project identifier is available." 
        )

        st.markdown("##### Time Assumptions")
        st.write(
            "Activities are analyzed using their **start year**. Ongoing activities without an end-year "
            "are treated as active for analytical purposes."
        )

        st.markdown("##### Budget Treatment")
        st.write(
            "Budget values represent **allocated funding**, not actual expenditure. "
            "Comparisons should be interpreted as financial exposure rather than performance."
        )

    with col_m2:
        st.markdown("##### Geographic Interpretation")
        st.write(
            "Analysis is based on the reported municipality. Data reflects **presence and concentration**, "
            "not necessarily beneficiary reach or outcomes."
        )

        st.markdown("##### Intended Use")
        st.write(
            "This dashboard supports **portfolio-level planning and coordination**. It identifies patterns and gaps "
            "to inform strategic discussions."
        )

        st.warning(
            "**Disclaimer:** Findings are based on administrative data and should not substitute "
            "formal monitoring, evaluation, or audit processes."
        )

# ==========================================
# TAB 1: DATA OVERVIEW
# ==========================================
with tabs[1]:
    st.subheader("Data Exploration & Summary")
    st.write("This tab provides a high-level summary and a searchable view of the raw activity records.")

    # --- TOP METRICS ---
    # We use 4 columns to give a quick "At a Glance" feel
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Total Records", len(filtered_df))
    with m2:
        st.metric("Municipalities", filtered_df["Municipality"].nunique())
    with m3:
        st.metric("Sectors", filtered_df["Sector"].nunique())
    with m4:
        # Formats the sum as a readable currency string
        total_budget = filtered_df['Budget'].sum()
        st.metric("Total Budget", f"${total_budget:,.0f}")

    st.divider()

    # --- INTERACTIVE DATA TABLE ---
    st.markdown("### Filtered Activity Records")
    st.caption("Use the dropdown below to add or remove columns from the view.")

    # Let users pick which columns they want to see, with a sensible default
    all_columns = filtered_df.columns.tolist()
    default_cols = [
        "Project_name",
        "Municipality",
        "Sector",
        "Budget",
        "Funding_Donor",
        "Implementing_Partner",
        "YearStart",
        "Status"
    ]
    
    # Intersection ensures we don't crash if a column name changes in your CSV
    valid_defaults = [c for c in default_cols if c in all_columns]

    selected_cols = st.multiselect(
        "Select variables to display:",
        options=all_columns,
        default=valid_defaults,
        key="data_overview_column_picker"
    )

    if selected_cols:
        st.dataframe(
            filtered_df[selected_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Please select at least one column to display the data table.")

# ==========================================
# TAB 2: PORTFOLIO OVERVIEW
# ==========================================
with tabs[2]:
    st.subheader("Portfolio Trends & Allocation")
    st.caption(
        "This overview highlights the scale of implementation and sectoral resource allocation. "
        "Comparing activity volume against budget helps identify capital-intensive sectors versus high-frequency interventions."
    )

    # 1. KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Activities", len(filtered_df))
    c2.metric("Municipalities", filtered_df["Municipality"].nunique())
    c3.metric("Sectors", filtered_df["Sector"].nunique())
    
    # Ongoing percentage calculation with safety check
    ongoing_val = filtered_df['Status'].eq('Ongoing - Active').mean() * 100 if not filtered_df.empty else 0
    c4.metric("% Ongoing", f"{ongoing_val:.1f}%")

    st.divider()

    # 2. Data Preparation for Charts
    # We aggregate here to keep the Altair code clean
    sector_stats = (
        filtered_df.groupby("Sector")
        .agg(
            Activities=("Sector", "count"),
            Total_Budget=("Budget", "sum")
        )
        .reset_index()
    )

    # 3. Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Activities by Sector")
        # Horizontal bar chart for readability with long sector names
        act_chart = (
            alt.Chart(sector_stats)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("Activities:Q", title="Number of Activities"),
                y=alt.Y("Sector:N", sort="-x", title=None),
                tooltip=["Sector", "Activities"]
            )
            .properties(height=500)
            .interactive()
        )
        st.altair_chart(act_chart, use_container_width=True)

    with col2:
        st.markdown("### Budget Allocation by Sector")
        budg_chart = (
            alt.Chart(sector_stats)
            .mark_bar(color="#F58518")
            .encode(
                x=alt.X("Total_Budget:Q", title="Total Budget ($)"),
                y=alt.Y("Sector:N", sort="-x", title=None),
                tooltip=[
                    "Sector", 
                    alt.Tooltip("Total_Budget:Q", format="$,.0f", title="Budget")
                ]
            )
            .properties(height=500)
            .interactive()
        )
        st.altair_chart(budg_chart, use_container_width=True)

# ==========================================
# TAB 3: GEOGRAPHIC FOOTPRINT
# ==========================================
with tabs[3]:
    st.subheader("Geographic Distribution of Activities")
    st.caption(
        "This analysis highlights where activities and funding are concentrated across municipalities. "
        "Discrepancies between activity counts and budget totals often indicate high-cost infrastructure "
        "vs. lower-cost community engagement programs."
    )

    # 1. Geographic Summary Calculation
    # We aggregate data specifically for this tab
    municipality_summary = (
        filtered_df.groupby("Municipality")
        .agg(
            Activities=("Municipality", "count"),
            Total_Budget=("Budget", "sum")
        )
        .reset_index()
    )

    # Calculate average budget per activity for each municipality
    # Using a fillna(0) to avoid NaN in calculations
    municipality_summary["Avg_Budget_per_Activity"] = (
        municipality_summary["Total_Budget"] / municipality_summary["Activities"]
    ).fillna(0)

    # 2. Key Performance Indicators (Geographic)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Municipalities Covered", filtered_df["Municipality"].nunique())
    with c2:
        st.metric("Total Activities", len(filtered_df))
    with c3:
        st.metric("Total Budget", f"${filtered_df['Budget'].sum():,.0f}")
    with c4:
        avg_overall = filtered_df['Budget'].mean() if not filtered_df.empty else 0
        st.metric("Avg Budget/Activity", f"${avg_overall:,.0f}")

    st.divider()

    # 3. Data Table View
    # Using st.column_config to format the currency directly in the table
    st.markdown("### Municipality Breakdown")
    st.dataframe(
        municipality_summary.sort_values("Total_Budget", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total_Budget": st.column_config.NumberColumn("Total Budget", format="$ %d"),
            "Avg_Budget_per_Activity": st.column_config.NumberColumn("Avg Budget", format="$ %d"),
        }
    )

    # 4. Comparative Charts
    col_geo1, col_geo2 = st.columns(2)

    with col_geo1:
        st.markdown("### Activity Concentration")
        geo_act_chart = (
            alt.Chart(municipality_summary)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("Activities:Q", title="Number of Activities"),
                y=alt.Y("Municipality:N", sort="-x", title=None),
                tooltip=["Municipality", "Activities"]
            )
            .properties(height=400)
        )
        st.altair_chart(geo_act_chart, use_container_width=True)

    with col_geo2:
        st.markdown("### Budget Allocation")
        geo_budg_chart = (
            alt.Chart(municipality_summary)
            .mark_bar(color="#F58518")
            .encode(
                x=alt.X("Total_Budget:Q", title="Total Budget ($)"),
                y=alt.Y("Municipality:N", sort="-x", title=None),
                tooltip=[
                    "Municipality", 
                    alt.Tooltip("Total_Budget:Q", format="$,.0f", title="Budget")
                ]
            )
            .properties(height=400)
        )
        st.altair_chart(geo_budg_chart, use_container_width=True)

# ==========================================
# TAB 4: SECTOR ALLOCATION
# ==========================================
with tabs[4]:
    st.subheader("Sectoral Reach & Capital Intensity")
    st.caption(
        "This section examines geographic reach and investment size. "
        "A sector with high 'Municipalities Covered' but low 'Avg Budget' suggests a wide-spread, community-level intervention model."
    )

    # 1. Aggregation for Sector Reach
    sector_summary = (
        filtered_df.groupby("Sector")
        .agg(
            Activities=("Sector", "count"),
            Total_Budget=("Budget", "sum"),
            Municipalities=("Municipality", "nunique")
        )
        .reset_index()
    )

    sector_summary["Avg_Budget_per_Activity"] = (
        sector_summary["Total_Budget"] / sector_summary["Activities"]
    ).fillna(0)

    # 2. KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Sectors", filtered_df["Sector"].nunique())
    with c2:
        st.metric("Most Active Sector", 
                  sector_summary.loc[sector_summary['Activities'].idxmax(), 'Sector'] if not sector_summary.empty else "N/A")
    with c3:
        st.metric("Total Budget", f"${filtered_df['Budget'].sum():,.0f}")
    with c4:
        avg_intense = sector_summary["Avg_Budget_per_Activity"].mean() if not sector_summary.empty else 0
        st.metric("Avg Sector Intensity", f"${avg_intense:,.0f}")

    st.divider()

    # 3. Visualizations
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("### Geographic Reach by Sector")
        reach_chart = (
            alt.Chart(sector_summary)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("Municipalities:Q", title="Municipalities Covered"),
                y=alt.Y("Sector:N", sort="-x", title=None),
                tooltip=["Sector", "Municipalities", "Activities"]
            )
            .properties(height=450)
        )
        st.altair_chart(reach_chart, use_container_width=True)

    with col_s2:
        st.markdown("### Capital Intensity (Avg Budget per Activity)")
        intensity_chart = (
            alt.Chart(sector_summary)
            .mark_bar(color="#76b7b2")
            .encode(
                x=alt.X("Avg_Budget_per_Activity:Q", title="Average Budget ($)"),
                y=alt.Y("Sector:N", sort="-x", title=None),
                tooltip=[
                    "Sector", 
                    alt.Tooltip("Avg_Budget_per_Activity:Q", format="$,.0f", title="Avg Budget")
                ]
            )
            .properties(height=450)
        )
        st.altair_chart(intensity_chart, use_container_width=True)

    # 4. Detailed Data Table
    with st.expander("View Full Sector Breakdown Table"):
        st.dataframe(
            sector_summary.sort_values("Activities", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total_Budget": st.column_config.NumberColumn("Total Budget", format="$ %d"),
                "Avg_Budget_per_Activity": st.column_config.NumberColumn("Avg Budget per Activity", format="$ %d"),
            }
        )

# ==========================================
# TAB 5: DONORS & IMPLEMENTERS
# ==========================================
with tabs[5]:
    st.subheader("Coordination Landscape: Donors & Partners")
    st.write(
        "This analysis highlights the distribution of funding and implementation load. "
        "It helps identify key partners and potential areas for increased coordination."
    )

    # 1. Data Preparation
    # Aggregation for Donors
    donor_summary = (
        filtered_df.groupby("Funding_Donor")
        .agg(
            Activities=("Funding_Donor", "count"),
            Total_Budget=("Budget", "sum")
        )
        .reset_index()
    )
    donor_summary["Avg_Budget_per_Activity"] = (
        donor_summary["Total_Budget"] / donor_summary["Activities"]
    ).fillna(0)

    # Aggregation for Implementers
    impl_summary = (
        filtered_df.groupby("Implementing_Partner")
        .agg(
            Activities=("Implementing_Partner", "count"),
            Total_Budget=("Budget", "sum"),
            Sectors=("Sector", "nunique")
        )
        .reset_index()
    )
    impl_summary["Avg_Budget_per_Activity"] = (
        impl_summary["Total_Budget"] / impl_summary["Activities"]
    ).fillna(0)

    # 2. KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Donors", filtered_df["Funding_Donor"].nunique())
    with c2:
        st.metric("Total Partners", filtered_df["Implementing_Partner"].nunique())
    with c3:
        top_donor = donor_summary.loc[donor_summary['Activities'].idxmax(), 'Funding_Donor'] if not donor_summary.empty else "N/A"
        st.metric("Lead Donor (Vol)", top_donor)
    with c4:
        top_impl = impl_summary.loc[impl_summary['Activities'].idxmax(), 'Implementing_Partner'] if not impl_summary.empty else "N/A"
        st.metric("Lead Partner (Vol)", top_impl)

    st.divider()

    # 3. Coordination Table with Toggle
    st.markdown("### Partner Coordination Table")
    view_option = st.radio(
        "View coordination by:",
        ["Donor", "Implementing Partner"],
        horizontal=True,
        key="coordination_toggle"
    )

    if view_option == "Donor":
        display_df = donor_summary.rename(columns={"Funding_Donor": "Entity"}).sort_values("Activities", ascending=False)
    else:
        display_df = impl_summary.rename(columns={"Implementing_Partner": "Entity"}).sort_values("Activities", ascending=False)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total_Budget": st.column_config.NumberColumn("Total Budget", format="$ %d"),
            "Avg_Budget_per_Activity": st.column_config.NumberColumn("Avg Budget/Act", format="$ %d"),
        }
    )

    # 4. Comparative Visualizations
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("### Top Donors by Activity Volume")
        donor_chart = (
            alt.Chart(donor_summary.nlargest(15, 'Activities'))
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("Activities:Q", title="Number of Activities"),
                y=alt.Y("Funding_Donor:N", sort="-x", title=None),
                tooltip=["Funding_Donor", "Activities", alt.Tooltip("Total_Budget:Q", format="$,.0f")]
            )
            .properties(height=400)
        )
        st.altair_chart(donor_chart, use_container_width=True)

    with col_d2:
        st.markdown("### Partner Implementation Intensity")
        impl_chart = (
            alt.Chart(impl_summary.nlargest(15, 'Avg_Budget_per_Activity'))
            .mark_bar(color="#76b7b2")
            .encode(
                x=alt.X("Avg_Budget_per_Activity:Q", title="Avg Budget per Activity ($)"),
                y=alt.Y("Implementing_Partner:N", sort="-x", title=None),
                tooltip=["Implementing_Partner", alt.Tooltip("Avg_Budget_per_Activity:Q", format="$,.0f")]
            )
            .properties(height=400)
        )
        st.altair_chart(impl_chart, use_container_width=True)


# ==========================================
# TAB 6: TIME DYNAMICS
# ==========================================
with tabs[6]:
    st.subheader("Temporal Trends: 2022–2025")
    st.write(
        "This section analyzes implementation timelines. By looking at the start years, we can track "
        "portfolio growth and the transition of projects from active to completed status."
    )

    # 1. Data Preparation for Time Trends
    # Basic aggregations
    activities_by_year = (
        filtered_df.groupby("YearStart")
        .size()
        .reset_index(name="Activities")
    )
    
    budget_by_year = (
        filtered_df.groupby("YearStart")["Budget"]
        .sum()
        .reset_index(name="Total_Budget")
    )

    status_by_year = (
        filtered_df.groupby(["YearStart", "Status"])
        .size()
        .reset_index(name="Count")
    )

    budget_muni_year = (
        filtered_df.groupby(["YearStart", "Municipality"])["Budget"]
        .sum()
        .reset_index()
    )

    # 2. KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Year Range", 
            f"{int(filtered_df['YearStart'].min())} – {int(filtered_df['YearStart'].max())}"
            if not filtered_df.empty else "N/A"
        )
    with c2:
        avg_act = activities_by_year["Activities"].mean() if not activities_by_year.empty else 0
        st.metric("Avg Activities / Year", f"{avg_act:.1f}")
    with c3:
        avg_bud = budget_by_year["Total_Budget"].mean() if not budget_by_year.empty else 0
        st.metric("Avg Budget / Year", f"${avg_bud:,.0f}")
    with c4:
        ongoing_pct = (filtered_df["Status"].eq("Ongoing - Active").mean() * 100) if not filtered_df.empty else 0
        st.metric("% Ongoing", f"{ongoing_pct:.1f}%")

    st.divider()

    # 3. Budget Heatmap (Municipality vs Time)
    st.markdown("### Budget Allocation Heatmap")
    st.caption("Darker blue indicates higher budget concentration in specific years and municipalities.")
    
    heatmap = (
        alt.Chart(budget_muni_year)
        .mark_rect()
        .encode(
            x=alt.X("YearStart:O", title="Year"),
            y=alt.Y("Municipality:N", title="Municipality"),
            color=alt.Color(
                "Budget:Q",
                scale=alt.Scale(scheme="blues"),
                title="Budget ($)"
            ),
            tooltip=[
                "Municipality",
                "YearStart",
                alt.Tooltip("Budget:Q", format="$,.0f")
            ]
        )
        .properties(height=400)
    )
    st.altair_chart(heatmap, use_container_width=True)

    # 4. Status and Volume Charts
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("### Activities Started per Year")
        act_year_chart = (
            alt.Chart(activities_by_year)
            .mark_bar(color="#4C78A8")
            .encode(
                x=alt.X("YearStart:O", title="Start Year"),
                y=alt.Y("Activities:Q", title="Activities"),
                tooltip=["YearStart", "Activities"]
            )
            .properties(height=350)
        )
        st.altair_chart(act_year_chart, use_container_width=True)

    with col_t2:
        st.markdown("### Project Status over Time")
        status_chart = (
            alt.Chart(status_by_year)
            .mark_bar()
            .encode(
                x=alt.X("YearStart:O", title="Start Year"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Ongoing - Active", "Completed"],
                        range=["#4C78A8", "#F58518"]
                    )
                ),
                tooltip=["YearStart", "Status", "Count"]
            )
            .properties(height=350)
        )
        st.altair_chart(status_chart, use_container_width=True)

# ==========================================
# TAB 7: INTERACTIVE MAP (FULL WIDTH)
# ==========================================
with tabs[7]:
    st.subheader("Geographic Distribution of Activities & Funding")

    # 0 National vs Sub-national (Always visible at the top) ---
    st.divider()
    st.markdown("### National  Portfolio")
    
    # Filtering for National (Improved keywords)
    is_national = filtered_df["Municipality"].str.contains("timor|leste|national|nationwide", case=False, na=False)
    national_df = filtered_df[is_national]
    
    c_nat1, c_nat2, c_nat3 = st.columns(3)
    c_nat1.metric("National Projects", len(national_df))
    c_nat2.metric("National Budget", f"${national_df['Budget'].sum():,.0f}")
    
    total_budget = filtered_df["Budget"].sum()
    ratio = (national_df["Budget"].sum() / total_budget * 100) if total_budget > 0 else 0
    c_nat3.metric("National Budget %", f"{ratio:.1f}%")

    st.divider()

    # --- 1. SINGLE COLUMN HEADER (Map Control) ---
    map_metric = st.selectbox(
        "Select Map Metric for Sub-national Visualization",
        ["Activities", "Total_Budget"],
        key="map_metric_selector"
    )
    st.caption("ℹ️ Map visualization excludes national-level records to focus on regional distribution. Click a municipality for details.")

    # 2. Data Preparation for Mapping
    geo_summary = (
        filtered_df.groupby(filtered_df["Municipality"].str.strip().str.upper())
        .agg(
            Activities=("Municipality", "count"),
            Total_Budget=("Budget", "sum")
        )
        .reset_index()
        .rename(columns={"Municipality": "mun_clean"})
    )

    # Merge activity data into GeoDataFrame
    map_df = gdf_raw.merge(geo_summary, on="mun_clean", how="left").fillna(0)

    # --- CRITICAL FIX FOR STREAMLIT CLOUD ERROR ---
    # Convert NumPy types (int64/float64) to standard Python types
    map_df["Activities"] = map_df["Activities"].astype(int)
    map_df["Total_Budget"] = map_df["Total_Budget"].astype(float)
    
    # Ensure any other numeric columns are standard floats
    for col in map_df.select_dtypes(include=['number']).columns:
        map_df[col] = map_df[col].astype(float)
    # ----------------------------------------------

    if map_metric == "Total_Budget":
        map_df["display_val"] = map_df[map_metric].apply(lambda x: f"${x:,.0f}")
    else:
        map_df["display_val"] = map_df[map_metric].astype(int).astype(str)

    # --- 3. DYNAMIC COLORING & LEGEND LOGIC ---
    zero_color = "#dddddd"
    active_colors = ["#c6dbef", "#6baed6", "#4292c6", "#2171b5", "#08306b"]
    pos_values = map_df[map_df[map_metric] > 0][map_metric]
    
    if not pos_values.empty:
        bins = np.linspace(pos_values.min(), pos_values.max(), len(active_colors) + 1).tolist()
    else:
        bins = [0] * (len(active_colors) + 1)

    def get_color(val):
        if val <= 0: return zero_color
        for i in range(len(active_colors)):
            if val <= bins[i+1]: return active_colors[i]
        return active_colors[-1]

    # --- 4. GENERATE MAP & ADD HTML LEGEND ---
    m = folium.Map(location=[-8.9, 126.5], zoom_start=8.5, tiles="cartodbpositron")

    # Define the Legend HTML
    legend_html = f'''
    <div style="
        position: fixed; 
        bottom: 30px; right: 30px; width: 200px; height: auto; 
        background-color: white; border:1.5px solid grey; z-index:9999; font-size:12px;
        padding: 10px; border-radius: 5px;
        ">
        <b>{map_metric.replace('_', ' ')}</b><br>
        <div style="margin-top: 5px;"><i style="background: {zero_color}; width: 14px; height: 14px; float: left; margin-right: 5px; opacity: 0.7;"></i> 0 / No Data</div>
    '''
    for i in range(len(active_colors)):
        low = f"{bins[i]:,.0f}"
        high = f"{bins[i+1]:,.0f}"
        legend_html += f'<div style="margin-top: 3px;"><i style="background: {active_colors[i]}; width: 14px; height: 14px; float: left; margin-right: 5px; opacity: 0.7;"></i> {low} - {high}</div>'
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))

    # Fix geometries before converting to GeoJSON
    map_df = map_df[map_df.geometry.notnull()]
    map_df = map_df.set_geometry("geometry")
    map_df = map_df.to_crs(epsg=4326)

    map_df["geometry"] = map_df["geometry"].buffer(0)

    folium.GeoJson(
        map_df.to_json(),
        style_function=lambda f: {
            "fillColor": get_color(f["properties"][map_metric]),
            "color": "white", "weight": 1, "fillOpacity": 0.8,
        },
        highlight_function=lambda x: {"weight": 3, "color": "#f39c12", "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=["adm1_name", "display_val"],
            aliases=["Municipality:", f"{map_metric.replace('_', ' ')}:"],
            sticky=True
        )
    ).add_to(m)

    # --- 5. RENDER FULL-WIDTH MAP ---
    map_output = st_folium(m, width=2000, height=600, key=f"full_map_{map_metric}")

    # --- 6. DETAIL VIEW ---
    if map_output and map_output.get("last_active_drawing"):
        props = map_output["last_active_drawing"]["properties"]
        muni_name = props['adm1_name']
        
        st.divider()
        st.subheader(f"Detailed Analysis: {muni_name}")
        
        muni_df = filtered_df[filtered_df["Municipality"].str.upper() == muni_name.upper()]
        
        d1, d2, d3 = st.columns(3)
        d1.metric("Activities", len(muni_df))
        d2.metric("Total Budget", f"${muni_df['Budget'].sum():,.0f}")
        
        # Safe mode calculation
        lead_partner = "N/A"
        if not muni_df.empty and not muni_df['Implementing_Partner'].isna().all():
            lead_partner = muni_df['Implementing_Partner'].mode()[0]
        d3.metric("Lead Partner", lead_partner)
        
        st.dataframe(muni_df[["Project_name", "Sector", "Budget", "Status"]], use_container_width=True, hide_index=True)


import plotly.graph_objects as go  # Move this to the top of your script

# ==========================================
# TAB 8: SDGs–NSDP ALIGNMENT
# ==========================================
with tabs[8]:
    st.subheader("Alignment with Strategic Frameworks")
    
    # --- 1. THE SANKEY DIAGRAM ---
    st.markdown("### Strategic Flow: NSDP to SDG Alignment")
    
    # Local cleaning for Sankey to prevent ghost labels
    sankey_raw = filtered_df.dropna(subset=['NSDP_Alignment', 'SDG_Alignment']).copy()
    
    if not sankey_raw.empty:
        # Normalize text to group correctly
        for col in ['NSDP_Alignment', 'SDG_Alignment']:
            sankey_raw[col] = sankey_raw[col].astype(str).str.strip()

        sankey_data = sankey_raw.groupby(['NSDP_Alignment', 'SDG_Alignment']).size().reset_index(name='value')
        
        # Define Nodes
        sources = sorted(sankey_data['NSDP_Alignment'].unique().tolist())
        targets = sorted(sankey_data['SDG_Alignment'].unique().tolist())
        all_nodes = sources + targets
        node_map = {name: i for i, name in enumerate(all_nodes)}
        
        sankey_data['source_idx'] = sankey_data['NSDP_Alignment'].map(node_map)
        sankey_data['target_idx'] = sankey_data['SDG_Alignment'].map(node_map)

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=40,
                thickness=15,
                line=dict(color="white", width=1),
                label=all_nodes,
                color="navy",
                customdata=all_nodes,
                hovertemplate='%{customdata}<extra></extra>'
            ),
            link=dict(
                source=sankey_data['source_idx'].tolist(),
                target=sankey_data['target_idx'].tolist(),
                value=sankey_data['value'].tolist(),
                color="rgba(173, 216, 230, 0.4)"
            )
        )])
        
        # FIXED: Font must be in update_layout, not in the node dictionary
        fig.update_layout(
            height=800,
            font=dict(color="black", size=12, family="Arial"),
            margin=dict(l=200, r=200, t=60, b=40),
            paper_bgcolor='white'
        )
        
        # Correctly removing shadows via trace configuration
        fig.update_traces(textfont_size=12, selector=dict(type='sankey'))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient alignment data to generate Sankey diagram.")

    st.divider()

    # --- 2. SDG-NSDP Correlation Heatmap ---
    st.markdown("### SDG–NSDP Alignment Matrix (Activity Count)")
    heatmap_cols = ["SDG_Alignment", "NSDP_Alignment"]
    if all(col in filtered_df.columns for col in heatmap_cols):
        heatmap_df = filtered_df.groupby(heatmap_cols).size().reset_index(name="project_count")
        if not heatmap_df.empty:
            heatmap = alt.Chart(heatmap_df).mark_rect().encode(
                x=alt.X("NSDP_Alignment:N", title="NSDP Pillar"),
                y=alt.Y("SDG_Alignment:N", title="SDG Alignment"),
                color=alt.Color("project_count:Q", scale=alt.Scale(scheme="purples")),
                tooltip=["SDG_Alignment", "NSDP_Alignment", "project_count"]
            ).properties(height=400)
            st.altair_chart(heatmap, use_container_width=True)
    
    st.divider()

    # --- 3. Transition Gap Analysis ---
    st.markdown("### Portfolio Maturity: Transition Gap Analysis")
    
    # Use .copy() to ensure we aren't modifying a slice of the original dataframe
    transition_df = filtered_df.copy()
    
    def get_transition_category(sector):
        s = str(sector).lower()
        if any(word in s for word in ["infra", "road", "power", "transport"]): return "Basic Infrastructure"
        if any(word in s for word in ["water", "health", "edu", "social"]): return "Basic Services"
        if any(word in s for word in ["econ", "farm", "agri", "trade", "finance"]): return "Economic Transition"
        return "Governance & Other"

    transition_df["transition_category"] = transition_df["Sector"].apply(get_transition_category)
    transition_summary = transition_df.groupby("transition_category").agg(
        total_budget=("Budget", "sum"), 
        count=("Budget", "count")
    ).reset_index()

    transition_chart = alt.Chart(transition_summary).mark_bar().encode(
        x=alt.X("total_budget:Q", title="Total Budget (USD)"),
        y=alt.Y("transition_category:N", sort="-x", title=None),
        color=alt.Color("transition_category:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=[alt.Tooltip("total_budget:Q", format="$,.0f"), "count"]
    ).properties(height=300)
    
    st.altair_chart(transition_chart, use_container_width=True)

# ==========================================
# TAB 9: MUNICIPALITY ANALYSIS
# ==========================================
with tabs[9]:
    st.subheader("Regional Deep-Dive & Strategic Flow")

    # 2. INDIVIDUAL MUNICIPALITY PROFILE
    st.markdown("### Individual Municipality Profile")
    muni_choice = st.selectbox("Select a Municipality to Profile:", sorted(filtered_df["Municipality"].unique()))
    
    muni_data = filtered_df[filtered_df["Municipality"] == muni_choice]
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.write(f"**Budget Allocation by Sector for {muni_choice}**")
        
        # Aggregated data for the bar chart
        muni_sector_summary = muni_data.groupby("Sector")["Budget"].sum().reset_index()
        
        # Horizontal Bar Chart (Easier to read than a pie chart)
        muni_bar = (
            alt.Chart(muni_sector_summary)
            .mark_bar(color="#76b7b2")
            .encode(
                x=alt.X("Budget:Q", title="Total Budget ($)"),
                y=alt.Y("Sector:N", sort="-x", title=None),
                tooltip=[
                    "Sector", 
                    alt.Tooltip("Budget:Q", format="$,.0f", title="Budget")
                ]
            )
            .properties(height=525)
        )
        st.altair_chart(muni_bar, use_container_width=True)

    with col_p2:
        st.write(f"**Key Statistics for {muni_choice}**")
        
        # Mini-metric row inside the column
        m1, m2, m3 = st.columns(3)
        m1.metric("Activities", len(muni_data))
        m2.metric("Total Budget", f"${muni_data['Budget'].sum():,.0f}")
        m3.metric("Partners", muni_data["Implementing_Partner"].nunique())
        
        st.write("**Top Projects in this Region**")
        st.dataframe(
            muni_data.sort_values("Budget", ascending=False)[["Project_name", "Sector", "Budget", "Status"]].head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Budget": st.column_config.NumberColumn(format="$ %d")
            }
        )





