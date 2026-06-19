import numpy as np
import pandas as pd
import scipy.stats as stats
import re
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

# ── 1. Ingestion & Preprocessing ──────────────────────────────────────────────
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "coffee.csv"))
df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

# Auto-detect Country Column names natively
country_col = "Location.Country" if "Location.Country" in df.columns else "Country" if "Country" in df.columns else None
if not country_col:
    country_matches = [c for c in df.columns if 'country' in c.lower()]
    country_col = country_matches[0] if country_matches else df.columns[0]

# Standardize the name inside our working dataframe and fix Tanzania casing
df["Location.Country"] = df[country_col].replace("Tanzania, United Republic Of", "Tanzania")

# Auto-detect Region Column (real data we actually have — no GPS needed)
region_col = "Location.Region" if "Location.Region" in df.columns else None
if not region_col:
    region_matches = [c for c in df.columns if 'region' in c.lower()]
    region_col = region_matches[0] if region_matches else None
if region_col:
    df["Location.Region"] = df[region_col].fillna("Unspecified region").astype(str).str.strip().str.title()
    df.loc[df["Location.Region"].str.lower().isin(["nan", ""]), "Location.Region"] = "Unspecified region"
else:
    df["Location.Region"] = "Unspecified region"

# Auto-detect Total Quality Score Column natively
total_score_col = "Data.Scores.Total" if "Data.Scores.Total" in df.columns else "Total.Cup.Points" if "Total.Cup.Points" in df.columns else "Total"
if total_score_col not in df.columns:
    score_matches = [c for c in df.columns if 'total' in c.lower() and 'score' in c.lower()]
    total_score_col = score_matches[0] if score_matches else df.columns[-1]
df["Data.Scores.Total"] = pd.to_numeric(df[total_score_col], errors="coerce")
df = df[df["Data.Scores.Total"].notna() & (df["Data.Scores.Total"] > 0)].copy()

# Auto-detect Production Bags Column natively
bags_col = "Data.Production.Number of bags" if "Data.Production.Number of bags" in df.columns else "Number.of.Bags" if "Number.of.Bags" in df.columns else "Bags"
df["Data.Production.Number of bags"] = pd.to_numeric(df[bags_col], errors="coerce") if bags_col in df.columns else np.nan
df["Data.Production.Number of bags"] = df["Data.Production.Number of bags"].fillna(0)

# Safe clean filtering for sensory metrics (only applied to columns that actually exist)
score_cols = ["Data.Scores.Aroma", "Data.Scores.Flavor", "Data.Scores.Aftertaste", "Data.Scores.Acidity", "Data.Scores.Body", "Data.Scores.Balance"]
cleanable_cols = [c for c in score_cols if c in df.columns]
if cleanable_cols:
    df = df[(df[cleanable_cols] > 0).all(axis=1)].copy()

# Auto-detect grower column identifiers
grower_col = "Data.Producer" if "Data.Producer" in df.columns else "Company" if "Company" in df.columns else "Owner"
if grower_col not in df.columns:
    df[grower_col] = "Lot #" + df.index.astype(str)
df[grower_col] = df[grower_col].fillna("Unknown Registered Grower")

# Average altitude column, if present, used purely for context (not for plotting position)
altitude_col = "Location.Altitude.Average" if "Location.Altitude.Average" in df.columns else None
if altitude_col:
    df["Location.Altitude.Average"] = pd.to_numeric(df[altitude_col], errors="coerce")

# Calculate Country aggregates for macro view mapping
country_summary = df.groupby("Location.Country").agg(
    Avg_Score=("Data.Scores.Total", "mean"),
    Total_Bags=("Data.Production.Number of bags", "sum"),
    Grower_Count=(grower_col, "nunique"),
    Lot_Count=("Data.Scores.Total", "count")
).reset_index()

# Calculate Region aggregates within each country — this is real, observed data,
# unlike fabricated GPS points, since the source file only records Country + Region text.
region_agg = {"Avg_Score": ("Data.Scores.Total", "mean"),
              "Lot_Count": ("Data.Scores.Total", "count"),
              "Total_Bags": ("Data.Production.Number of bags", "sum")}
if altitude_col:
    region_agg["Avg_Altitude"] = ("Location.Altitude.Average", "mean")

region_summary = df.groupby(["Location.Country", "Location.Region"]).agg(**region_agg).reset_index()

# ── 2. Initialize the Dash Application Framework ──────────────────────────────
app = dash.Dash(__name__)
server = app.server  # exposed so a production server (gunicorn) can run this

app.layout = html.Div(
    style={'backgroundColor': '#0f1117', 'color': '#aaaaaa', 'fontFamily': 'sans-serif', 'padding': '20px', 'minHeight': '100vh'},
    children=[
        html.Div([
            html.H1("Global Coffee Provenance Explorer", style={'color': 'white', 'fontWeight': 'bold', 'margin': '0 0 5px 0', 'fontSize': '28px'}),
            html.P("Geographic drill-down pipeline mapping country-level quality metrics down to named growing regions.", style={'color': '#aaaaaa', 'margin': '0 0 20px 0', 'fontSize': '14px'}),
        ], style={'borderBottom': '1px solid #2a2d3a', 'paddingBottom': '15px'}),

        html.Div([
            html.Div([
                html.H3("Navigation & Control Center", style={'color': 'white', 'fontSize': '16px', 'marginTop': '0'}),
                html.Label("Select Target Country Baseline:", style={'fontSize': '12px', 'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': c, 'value': c} for c in sorted(df["Location.Country"].unique())],
                    placeholder="Global Overview Mode (Click map or dropdown to zoom)",
                    style={'backgroundColor': '#1a1d27', 'color': 'black', 'borderRadius': '4px'}
                ),
                html.Div(id='inspector-panel', style={'marginTop': '25px', 'padding': '15px', 'backgroundColor': '#1a1d27', 'borderRadius': '6px', 'border': '1px solid #2a2d3a'})
            ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

            html.Div([
                dcc.Graph(id='provenance-map', style={'height': '750px', 'borderRadius': '6px', 'overflow': 'hidden'})
            ], style={'width': '72%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'display': 'flex', 'marginTop': '20px'})
    ]
)

# ── 3. Dynamic Interactive Callbacks ──────────────────────────────────────────
@app.callback(
    [Output('provenance-map', 'figure'),
     Output('inspector-panel', 'children'),
     Output('country-dropdown', 'value')],
    [Input('country-dropdown', 'value'),
     Input('provenance-map', 'clickData')]
)
def update_view_and_inspector(selected_country, clickData):
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id if ctx.triggered else None

    clicked_region = None

    # Handle clicks coming from the visual itself
    if triggered_id == 'provenance-map' and clickData:
        point_data = clickData['points'][0]
        if 'location' in point_data:
            # Choropleth country region clicked -> drill into that country
            selected_country = point_data['location']
        elif 'x' in point_data and selected_country:
            # Region bar clicked while already inside a country -> show region detail
            clicked_region = point_data['x']

    # 🗺️ VIEW A: High-Level Global Overview (choropleth, real country-level averages)
    if not selected_country:
        fig = px.choropleth(
            country_summary,
            locations="Location.Country",
            locationmode="country names",
            color="Avg_Score",
            color_continuous_scale="Purples",
            labels={'Avg_Score': 'Mean Quality'},
            scope="world"
        )
        fig.update_layout(
            geo=dict(bgcolor='#1a1d27', showframe=False, showcoastlines=True, projection_type='equirectangular'),
            template="plotly_dark",
            paper_bgcolor="#1a1d27",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        inspector_content = html.Div([
            html.H4("Global Ingestion Profile", style={'color': 'white', 'margin': '0 0 10px 0'}),
            html.P("Click a country region directly on the world map, or choose from the selection menu, to break that country down by growing region.")
        ])
        return fig, inspector_content, None

    # 🗺️ VIEW B: Country selected -> bar chart of real growing regions within it
    country_df = df[df["Location.Country"] == selected_country]
    country_regions = region_summary[region_summary["Location.Country"] == selected_country].sort_values("Avg_Score", ascending=False)
    matching_meta = country_summary[country_summary["Location.Country"] == selected_country]

    if country_df.empty or matching_meta.empty:
        return px.bar(title="No records match bounds."), html.P("No records match bounds."), selected_country

    country_meta = matching_meta.iloc[0]

    fig = px.bar(
        country_regions,
        x="Location.Region",
        y="Avg_Score",
        color="Avg_Score",
        color_continuous_scale="Viridis",
        labels={'Avg_Score': 'Avg Total Score', 'Location.Region': 'Growing Region'},
        hover_data={"Lot_Count": True, "Avg_Score": ":.2f"}
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        margin={"r": 20, "t": 10, "l": 40, "b": 120},
        xaxis_tickangle=-45,
        yaxis_range=[max(0, country_regions["Avg_Score"].min() - 2), country_regions["Avg_Score"].max() + 2]
    )

    if clicked_region:
        region_row = country_regions[country_regions["Location.Region"] == clicked_region]
        if not region_row.empty:
            r = region_row.iloc[0]
            detail_rows = [
                html.H4(f"{clicked_region}", style={'color': '#e8c468', 'margin': '0 0 10px 0'}),
                html.P([html.Strong("Country: "), selected_country]),
                html.P([html.Strong("Avg Total Score: "), f"{r['Avg_Score']:.2f} pts"]),
                html.P([html.Strong("Lots Recorded: "), f"{int(r['Lot_Count'])}"]),
            ]
            if "Avg_Altitude" in r and pd.notna(r["Avg_Altitude"]):
                detail_rows.append(html.P([html.Strong("Avg Altitude: "), f"{r['Avg_Altitude']:.0f} m"]))
            inspector_content = html.Div(detail_rows)
            return fig, inspector_content, selected_country

    inspector_content = html.Div([
        html.H4(f"Origin Report: {selected_country}", style={'color': '#5b8db8', 'margin': '0 0 10px 0'}),
        html.P([html.Strong("Cleaned Lot Population: "), f"{len(country_df)} rows"]),
        html.P([html.Strong("Unique Registered Growers: "), f"{country_meta['Grower_Count']}"]),
        html.P([html.Strong("Distinct Named Regions: "), f"{country_regions['Location.Region'].nunique()}"]),
        html.P([html.Strong("National Quality Average: "), f"{country_meta['Avg_Score']:.2f} pts"]),
        html.P([html.Strong("Aggregate Production Yield: "), f"{int(country_meta['Total_Bags']):,} Bags"]),
        html.Hr(style={'borderColor': '#2a2d3a', 'margin': '15px 0'}),
        html.P("💡 Click any bar to inspect that region's sensory profile in detail.", style={'fontSize': '11px', 'fontStyle': 'italic'})
    ])
    return fig, inspector_content, selected_country


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
