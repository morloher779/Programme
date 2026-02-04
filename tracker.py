import streamlit as st
import osmnx as ox
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------------------------------------------------
# 1. KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Tracker (Google Sheets)", layout="wide")

PLACE_NAME = "Niederaichbach, Germany"
SHEET_NAME = "Flyer_Daten"  # <--- HIER NAME DEINER GOOGLE TABELLE
ADMIN_PASSWORD = "admin"

RELEVANT_HIGHWAYS = [
    "residential", "living_street", "service", "secondary", "tertiary", 
    "unclassified", "primary", "track"
]

# ---------------------------------------------------------
# 2. GOOGLE SHEETS VERBINDUNG
# ---------------------------------------------------------

def get_google_sheet():
    """Verbindet sich mit der Google Tabelle (Modern & Stabil)"""
    try:
        # 1. Credentials laden
        # Wir greifen auf die secrets zu und wandeln sie in ein normales Dictionary um
        creds_dict = dict(st.secrets["gcp_service_account"])
        
        # 2. Fix für Zeilenumbrüche (falls nötig)
        if "\\n" in creds_dict["private_key"]:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            
        # 3. Verbindung aufbauen (Direkt mit gspread, ohne oauth2client)
        # gspread service_account_from_dict ist viel robuster
        client = gspread.service_account_from_dict(creds_dict)
        
        # 4. Tabelle öffnen
        sheet = client.open(SHEET_NAME).sheet1
        return sheet
        
    except KeyError as e:
        st.error(f"FEHLER IN SECRETS.TOML: Der Schlüssel '{e}' fehlt in der Datei!")
        st.stop()
    except Exception as e:
        st.error(f"Verbindungsfehler: {e}")
        st.stop()

def load_progress_google():
    """Lädt die Liste aus Spalte A der Tabelle."""
    sheet = get_google_sheet()
    # Alle Werte aus Spalte 1 (A) holen
    data = sheet.col_values(1)
    return data

def save_progress_google(completed_list):
    """Löscht Spalte A und schreibt die neue Liste rein."""
    sheet = get_google_sheet()
    sheet.clear() # Alles löschen
    # Liste in Spalte A schreiben (gspread braucht eine Liste von Listen für Spalten)
    # Format: [['Strasse A'], ['Strasse B'], ...]
    column_data = [[street] for street in completed_streets]
    if column_data:
        sheet.update(range_name='A1', values=column_data)
    else:
        pass # Nichts zu tun wenn leer

# ---------------------------------------------------------
# 3. KARTENDATEN LADEN
# ---------------------------------------------------------

@st.cache_data
def load_map_data():
    tags_streets = {"highway": RELEVANT_HIGHWAYS}
    gdf_streets = ox.features_from_place(PLACE_NAME, tags=tags_streets)
    gdf_streets_proj = gdf_streets.to_crs(gdf_streets.estimate_utm_crs())
    
    if 'name' in gdf_streets_proj.columns:
        gdf_streets_proj = gdf_streets_proj[gdf_streets_proj['name'].notna()]
    gdf_streets_proj = gdf_streets_proj[['name', 'geometry']]

    tags_buildings = {"building": True}
    gdf_buildings = ox.features_from_place(PLACE_NAME, tags=tags_buildings)
    gdf_buildings_proj = gdf_buildings.to_crs(gdf_streets_proj.crs)
    gdf_buildings_proj['centroid'] = gdf_buildings_proj.geometry.centroid
    
    gdf_points = gpd.GeoDataFrame(geometry=gdf_buildings_proj['centroid'], crs=gdf_buildings_proj.crs)
    joined = gpd.sjoin_nearest(gdf_points, gdf_streets_proj, distance_col="dist")
    counts = joined.groupby('name').size().reset_index(name='house_count')
    
    gdf_streets_final = gdf_streets_proj.merge(counts, on='name', how='left')
    gdf_streets_final['house_count'] = gdf_streets_final['house_count'].fillna(0).astype(int)
    
    return gdf_streets_final.to_crs("EPSG:4326")

# ---------------------------------------------------------
# 4. DASHBOARD UI
# ---------------------------------------------------------

st.title("☁️ Online Straßen-Tracker")

with st.spinner("Lade Karte..."):
    gdf_streets = load_map_data()

all_street_names = sorted(gdf_streets['name'].unique().tolist())

# --- GOOGLE LOAD ---
# Wir laden die Daten nur am Anfang oder wenn wir speichern
if 'completed_streets' not in st.session_state:
    with st.spinner("Lade Daten von Google Sheets..."):
        st.session_state.completed_streets = load_progress_google()

completed_streets = st.session_state.completed_streets

# --- SIDEBAR ---
st.sidebar.header("Fortschritt melden")
open_streets = sorted([s for s in all_street_names if s not in completed_streets])

newly_done = st.sidebar.multiselect(
    "Straßen als erledigt markieren:", 
    options=open_streets,
    placeholder="Wähle eine Straße..."
)

# SPEICHERN LOGIK
if newly_done:
    updated_list = sorted(list(set(completed_streets + newly_done)))
    
    with st.spinner("Speichere in Google Cloud..."):
        save_progress_google(updated_list)
        st.session_state.completed_streets = updated_list
        st.rerun()

# --- ADMIN ---
st.sidebar.markdown("---")
with st.sidebar.expander("Admin: Fehler korrigieren"):
    if st.text_input("Passwort", type="password") == ADMIN_PASSWORD:
        to_remove = st.multiselect("Lösche Erledigte:", options=completed_streets)
        if st.button("Löschen anwenden"):
            # Entferne die ausgewählten aus der Liste
            new_list = [s for s in completed_streets if s not in to_remove]
            save_progress_google(new_list)
            st.session_state.completed_streets = new_list
            st.rerun()

# --- STATISTIK & KARTE ---
total_houses = gdf_streets.drop_duplicates(subset=['name'])['house_count'].sum()
unique_streets = gdf_streets.drop_duplicates(subset=['name'])
done_houses = unique_streets[unique_streets['name'].isin(completed_streets)]['house_count'].sum()
percent = int((done_houses / total_houses) * 100) if total_houses > 0 else 0

col1, col2 = st.columns(2)
col1.metric("Erledigte Haushalte", f"{done_houses} / {total_houses}")
col2.metric("Fortschritt", f"{percent}%")
st.progress(percent)

center_lat = gdf_streets.geometry.centroid.y.mean()
center_lon = gdf_streets.geometry.centroid.x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

def style_function(feature):
    name = feature['properties']['name']
    house_cnt = feature['properties']['house_count']
    if name in completed_streets:
        return {'color': '#2ecc71', 'weight': 5, 'opacity': 0.8}
    else:
        return {'color': '#bdc3c7', 'weight': 2, 'opacity': 0.5} if house_cnt == 0 else {'color': '#e74c3c', 'weight': 4, 'opacity': 0.7}

tooltip = folium.GeoJsonTooltip(fields=["name", "house_count"], aliases=["Straße:", "Haushalte:"], localize=True)
folium.GeoJson(gdf_streets, style_function=style_function, tooltip=tooltip, name="Straßennetz").add_to(m)

st_folium(m, width=None, height=600)