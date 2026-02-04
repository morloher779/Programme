import osmnx as ox
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans
import folium
import numpy as np
from shapely.geometry import Point
import webbrowser
import os
import simplekml
from shapely.geometry import MultiPoint, Polygon
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ---------------------------------------------------------
# 1. KONFIGURATION & DATEN
# ---------------------------------------------------------

# Ort für OpenStreetMap
place_name = "Niederaichbach, Germany"

# Die Koordinaten der Austräger (aus deiner Nachricht)
austraeger = {
    "Ben": (48.61301383421625, 12.315285732488933),
    "Stefan": (48.613981693021756, 12.316412026926503),
    "Tina": (48.61098694593706, 12.31425752002685),
    "Flo": (48.612898027022375, 12.3238266361659655),
    "Jonas & Resi": (48.614478128541556, 12.324610755217856),
    "MJ": (48.61296334932955, 12.324970848795243),
    "Tom": (48.6107967419515, 12.317408086973165),
    "Vale": (48.610198660058224, 12.318294561992476),
    "Nico": (48.61284435799261, 12.320471043760076)
}

n_puzzle_pieces = len(austraeger) * 6 

# Farben
html_colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'darkred', 
    'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue'
]
kml_colors = [
    simplekml.Color.red, simplekml.Color.blue, simplekml.Color.green, simplekml.Color.purple,
    simplekml.Color.orange, simplekml.Color.darkred, simplekml.Color.lightcoral, simplekml.Color.yellow,
    simplekml.Color.darkblue, simplekml.Color.darkgreen, simplekml.Color.cyan
]

print(f"--- Starte 'Puzzle-Methode' (Fair & Logisch) ---")

# ---------------------------------------------------------
# 2. DATEN LADEN
# ---------------------------------------------------------
print("1/5: Lade Gebäudedaten...")

try:
    tags = {"building": True}
    gdf = ox.features_from_place(place_name, tags=tags)
    
    if 'addr:street' not in gdf.columns:
        gdf['addr:street'] = "Unbekannt"
    else:
        gdf['addr:street'] = gdf['addr:street'].fillna("Unbekannt")

    estimated_crs = gdf.estimate_utm_crs()
    target_crs = estimated_crs if estimated_crs is not None else "EPSG:32633"
    gdf_proj = gdf.to_crs(target_crs)
    gdf_proj['centroid'] = gdf_proj.geometry.centroid
    X = np.column_stack((gdf_proj['centroid'].x, gdf_proj['centroid'].y))
    
    total_houses = len(X)
    target_per_person = total_houses / len(austraeger)
    print(f"     -> {total_houses} Gebäude.")
    print(f"     -> Ziel: Ca. {int(target_per_person)} Häuser pro Person.")

except Exception as e:
    print(f"FEHLER: {e}")
    exit()

# ---------------------------------------------------------
# 3. PUZZLE-STÜCKE ERSTELLEN (Micro-Clustering)
# ---------------------------------------------------------
print(f"2/5: Zerschneide Ort in {n_puzzle_pieces} kleine Blöcke...")

# Wir nutzen wieder Ward, aber diesmal für viel mehr Cluster als Personen.
# Das erzeugt kleine, kompakte Nachbarschaften (z.B. 1-2 Straßenzüge).
ward = AgglomerativeClustering(n_clusters=n_puzzle_pieces, linkage='ward')
micro_labels = ward.fit_predict(X)

# Informationen über jedes Puzzle-Teil sammeln
puzzle_pieces = []
for i in range(n_puzzle_pieces):
    mask = (micro_labels == i)
    points = X[mask]
    center = points.mean(axis=0)
    size = len(points)
    puzzle_pieces.append({
        'id': i,
        'center': center,
        'size': size,
        'assigned': False
    })

# ---------------------------------------------------------
# 4. FAIRE VERTEILUNG (Greedy Balancing)
# ---------------------------------------------------------
print("3/5: Verteile Puzzleteile fair an Personen...")

person_names = list(austraeger.keys())
person_load = {name: 0 for name in person_names} # Wie viele Häuser hat jeder schon?
person_assignments = {name: [] for name in person_names} # Welche Puzzle-IDs hat jeder?

# Personen-Koordinaten projizieren
df_pers = pd.DataFrame(austraeger.values(), columns=['lat', 'lon'], index=person_names)
gdf_pers = gpd.GeoDataFrame(df_pers, geometry=gpd.points_from_xy(df_pers.lon, df_pers.lat), crs="EPSG:4326")
gdf_pers_proj = gdf_pers.to_crs(target_crs)
person_coords_proj = np.column_stack((gdf_pers_proj.geometry.x, gdf_pers_proj.geometry.y))

# Solange noch Puzzle-Teile da sind...
unassigned_count = n_puzzle_pieces

while unassigned_count > 0:
    # 1. Finde die Person, die aktuell am WENIGSTEN Häuser hat (Priorität für Fairness)
    # Sortiere nach: 1. Load (aufsteigend), 2. Zufall (um Deadlocks zu vermeiden)
    sorted_persons = sorted(person_names, key=lambda p: person_load[p])
    current_person = sorted_persons[0]
    p_idx = person_names.index(current_person)
    p_center = person_coords_proj[p_idx]
    
    # 2. Finde das nächste verfügbare Puzzle-Teil für diese Person
    best_piece = None
    best_dist = float('inf')
    
    for piece in puzzle_pieces:
        if not piece['assigned']:
            # Distanz berechnen
            dist = np.linalg.norm(piece['center'] - p_center)
            if dist < best_dist:
                best_dist = dist
                best_piece = piece
    
    # 3. Zuweisen
    if best_piece:
        best_piece['assigned'] = True
        person_assignments[current_person].append(best_piece['id'])
        person_load[current_person] += best_piece['size']
        unassigned_count -= 1
        
        # Kleiner Trick: Den "Mittelpunkt" der Person verschieben wir leicht in Richtung
        # des neuen Gebiets, damit das nächste Puzzleteil daran anschließt (Wachstum).
        # Wir gewichten den neuen Punkt mit 10%, damit er nicht wild springt.
        person_coords_proj[p_idx] = (person_coords_proj[p_idx] * 0.9) + (best_piece['center'] * 0.1)

# Mapping zurück auf die Häuser übertragen
# Wir bauen ein Array, das für jeden Micro-Cluster sagt, wem er gehört
micro_to_person = {}
for name, piece_ids in person_assignments.items():
    for pid in piece_ids:
        micro_to_person[pid] = name

# Zuweisung im DataFrame speichern
gdf_proj['micro_id'] = micro_labels
gdf_proj['assigned_person'] = gdf_proj['micro_id'].map(micro_to_person)

# ---------------------------------------------------------
# 5. VISUALISIERUNG
# ---------------------------------------------------------
print("4/5: Erstelle Karten...")

gdf_final = gdf_proj.to_crs("EPSG:4326")
center_lat = gdf_final.geometry.centroid.y.mean()
center_lon = gdf_final.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
kml = simplekml.Kml()

# Iterieren über NAMEN für konstante Farben
for idx, name in enumerate(person_names):
    c_html = html_colors[idx % len(html_colors)]
    c_kml = kml_colors[idx % len(kml_colors)]
    
    subset = gdf_final[gdf_final['assigned_person'] == name]
    
    # HTML
    for _, row in subset.iterrows():
        geom = row.geometry
        lat, lon = (geom.centroid.y, geom.centroid.x) if geom.geom_type in ['Polygon', 'MultiPolygon'] else (geom.y, geom.x)
        folium.CircleMarker(
            location=(lat, lon), radius=3, color=c_html,
            fill=True, fill_opacity=0.8, popup=name, stroke=False
        ).add_to(m)

    start_c = austraeger[name]
    folium.Marker(start_c, popup=f"START: {name}", icon=folium.Icon(color=c_html, icon='home')).add_to(m)

    # KML
    fol = kml.newfolder(name=name)
    # Umrisse generieren: Da wir jetzt "Puzzle-Teile" haben, kann ein Gebiet auch mal aus
    # zwei getrennten Blöcken bestehen. Wir versuchen trotzdem eine Hülle.
    if len(subset) >= 3:
        points_list = subset.geometry.centroid.tolist()
        hull = MultiPoint(points_list).convex_hull
        if isinstance(hull, Polygon):
            coords = list(hull.exterior.coords)
            pol = fol.newpolygon(name=f"Gebiet {name}")
            pol.outerboundaryis = coords
            pol.style.polystyle.color = simplekml.Color.changealphaint(110, c_kml)
            pol.style.linestyle.color = c_kml
            pol.style.linestyle.width = 3
        else:
             for _, row in subset.iterrows():
                g = row.geometry.centroid
                p = fol.newpoint(name="Haus", coords=[(g.x, g.y)])
                p.style.iconstyle.color = c_kml
    pnt = fol.newpoint(name=f"Start {name}", coords=[(start_c[1], start_c[0])])

m.save("Gebietsverteilung_Fair.html")
kml.save("Gebietsverteilung_Fair.kml")

# ---------------------------------------------------------
# 6. STATISTIK & TEXT OUTPUT
# ---------------------------------------------------------
print("5/5: Erstelle Statistik...")

with open("Verteilung_Fairness_Check.txt", "w", encoding="utf-8") as f:
    f.write("FAIRNESS CHECK (PUZZLE METHODE)\n")
    f.write("===============================\n")
    f.write(f"Gesamtgebäude: {total_houses}\n")
    f.write(f"Idealwert pro Person: {total_houses / len(austraeger):.1f}\n\n")
    
    sorted_people = sorted(person_names, key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else x)
    
    for name in sorted_people:
        count = person_load[name]
        diff = count - target_per_person
        diff_str = f"+{diff:.0f}" if diff > 0 else f"{diff:.0f}"
        
        f.write(f"--- {name}: {count} Gebäude (Abweichung: {diff_str}) ---\n")
        
        # Straßen-Liste
        subset = gdf_final[gdf_final['assigned_person'] == name]
        streets = sorted([s for s in subset['addr:street'].unique() if s and s != "Unbekannt"])
        if streets:
             f.write(f"Straßen (Auszug): {', '.join(streets[:5])}...\n")
        f.write("\n")

print("-" * 30)
print("FERTIG!")
print("1. Gebietsverteilung_Fair.html")
print("2. Gebietsverteilung_Fair.kml")
print("3. Verteilung_Fairness_Check.txt")
print("-" * 30)

webbrowser.open("Verteilung_Fairness_Check.txt")