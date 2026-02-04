import streamlit as st
import osmnx as ox
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import os

# ---------------------------------------------------------
# 1. KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Straßen-Tracker (Linien)", layout="wide")

PLACE_NAME = "Niederaichbach, Germany"
DATA_FILE = "fortschritt_strassen.json"

# Welche Straßentypen wollen wir sehen? (Autobahnen etc. rausfiltern)
RELEVANT_HIGHWAYS = [
    "residential", "living_street", "service", "secondary", "tertiary", 
    "unclassified", "primary", "track"
]

# ---------------------------------------------------------
# 2. INTELLIGENTE DATENVERARBEITUNG
# ---------------------------------------------------------

@st.cache_data
def load_and_match_data():
    """
    Lädt Straßen und Häuser. 
    Ordnet jedes Haus der am nächsten liegenden Straße zu.
    Gibt ein GeoDataFrame der Straßen zurück, inkl. 'house_count'.
    """
    
    # A) STRASSEN LADEN (Linien)
    # Wir holen nur relevante Wege, wo auch Häuser stehen könnten
    tags_streets = {"highway": RELEVANT_HIGHWAYS}
    gdf_streets = ox.features_from_place(PLACE_NAME, tags=tags_streets)
    
    # Wichtig: Wir brauchen eine saubere Projektion für Distanzmessung (Meter)
    gdf_streets_proj = gdf_streets.to_crs(gdf_streets.estimate_utm_crs())
    
    # Filtern: Nur Straßen, die einen Namen haben
    if 'name' in gdf_streets_proj.columns:
        gdf_streets_proj = gdf_streets_proj[gdf_streets_proj['name'].notna()]
    
    # Wir behalten nur die Spalten 'name' und 'geometry'
    gdf_streets_proj = gdf_streets_proj[['name', 'geometry']]

    # B) HÄUSER LADEN (Punkte)
    tags_buildings = {"building": True}
    gdf_buildings = ox.features_from_place(PLACE_NAME, tags=tags_buildings)
    gdf_buildings_proj = gdf_buildings.to_crs(gdf_streets_proj.crs)
    
    # Wir nehmen den Mittelpunkt der Häuser für die Berechnung
    gdf_buildings_proj['centroid'] = gdf_buildings_proj.geometry.centroid
    
    # C) SPATIAL JOIN (Die Magie: Welches Haus gehört zu welcher Straße?)
    # Wir erstellen ein temporäres GDF nur mit Punkten, um sjoin_nearest zu nutzen
    gdf_points = gpd.GeoDataFrame(
        geometry=gdf_buildings_proj['centroid'], 
        crs=gdf_buildings_proj.crs
    )
    
    # Finde für jeden Punkt die nächste Straße
    # 'sjoin_nearest' gibt uns für jedes Haus den Index der Straße zurück
    joined = gpd.sjoin_nearest(gdf_points, gdf_streets_proj, distance_col="dist")
    
    # D) ZÄHLEN: Wie viele Häuser pro Straßenname?
    # Wir gruppieren nach dem Straßennamen aus dem Join
    counts = joined.groupby('name').size().reset_index(name='house_count')
    
    # E) DATEN ZUSAMMENFÜHREN
    # Wir hängen die Haus-Anzahl an die Original-Straßen
    # Achtung: Eine Straße kann aus mehreren Segmenten bestehen. 
    # Wir wollen aber die Summe der Häuser für den Namen wissen.
    gdf_streets_final = gdf_streets_proj.merge(counts, on='name', how='left')
    
    # Leere Auffüllen (Straßen ohne Häuser sind trotzdem Straßen)
    gdf_streets_final['house_count'] = gdf_streets_final['house_count'].fillna(0).astype(int)
    
    # Zurück zu GPS Koordinaten für Folium
    return gdf_streets_final.to_crs("EPSG:4326")

def load_progress():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_progress(completed_list):
    with open(DATA_FILE, "w") as f:
        json.dump(completed_list, f)

# ---------------------------------------------------------
# 3. DASHBOARD
# ---------------------------------------------------------

st.title("🛣️ Straßen-Tracker (Linien-Modus)")

with st.spinner("Berechne Zuordnung Häuser -> Straßen... (kann kurz dauern)"):
    gdf_streets = load_and_match_data()

# Liste aller einzigartigen Straßennamen holen
all_street_names = sorted(gdf_streets['name'].unique().tolist())

# Fortschritt laden
completed_streets = load_progress()

# --- SIDEBAR ---
st.sidebar.header("Straßen abhaken")

selected_streets = st.sidebar.multiselect(
    "Suche Straße:", 
    options=all_street_names,
    default=[s for s in completed_streets if s in all_street_names],
    help="Tippe den Namen ein. Die ganze Straße wird grün markiert."
)

# Speichern
if set(selected_streets) != set(completed_streets):
    save_progress(selected_streets)
    st.rerun()

# --- STATISTIK ---
# Wir berechnen die erledigten HÄUSER basierend auf den erledigten STRASSEN
total_houses = gdf_streets.drop_duplicates(subset=['name'])['house_count'].sum()

# Erledigte Straßen filtern (Achtung: Duplikate entfernen, da Straße aus mehreren Linien bestehen kann)
unique_streets = gdf_streets.drop_duplicates(subset=['name'])
done_houses = unique_streets[unique_streets['name'].isin(selected_streets)]['house_count'].sum()

if total_houses > 0:
    percent = int((done_houses / total_houses) * 100)
else:
    percent = 0

col1, col2 = st.columns(2)
col1.metric("Erledigte Haushalte (geschätzt)", f"{done_houses} / {total_houses}")
col2.metric("Fortschritt", f"{percent}%")
st.progress(percent)

# --- KARTE ---
st.subheader("Übersichtskarte")

# Zentrum berechnen
center_lat = gdf_streets.geometry.centroid.y.mean()
center_lon = gdf_streets.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

# Wir definieren eine Style-Funktion für die Linien
# Das ist viel schneller als jede Linie einzeln in einer Schleife zu adden
def style_function(feature):
    name = feature['properties']['name']
    house_cnt = feature['properties']['house_count']
    
    # Farbe bestimmen
    if name in selected_streets:
        return {'color': '#2ecc71', 'weight': 5, 'opacity': 0.8} # Grün, dick
    else:
        # Wenn die Straße 0 Häuser hat (z.B. Feldweg), machen wir sie grau/dünn
        if house_cnt == 0:
            return {'color': '#bdc3c7', 'weight': 2, 'opacity': 0.5} # Grau
        else:
            return {'color': '#e74c3c', 'weight': 4, 'opacity': 0.7} # Rot

# Tooltip (Was passiert beim Mouseover)
tooltip = folium.GeoJsonTooltip(
    fields=["name", "house_count"],
    aliases=["Straße:", "Haushalte ca.:"],
    localize=True,
    sticky=False
)

# Layer hinzufügen
folium.GeoJson(
    gdf_streets,
    style_function=style_function,
    tooltip=tooltip,
    name="Straßennetz"
).add_to(m)

st_folium(m, width=None, height=600)