import pandas as pd
import numpy as np
import folium
from pathlib import Path
import re

# Create output directory for maps
output_dir = Path('analysis_output/geospatial_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print("Loading data...")
df = pd.read_csv('mo.csv')

# Ensure Zipcode is a string with 5 digits
df['Zipcode'] = df['Zipcode'].astype(str).str.zfill(5)

# Convert columns to numeric where possible
for col in df.columns:
    if col != 'Zipcode':
        try:
            # Remove commas and convert to numeric
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        except:
            pass

# Print column info to verify data
print(f"Columns in dataset: {df.columns.tolist()}")
print(f"Data types: {df.dtypes}")

# Check if we have the expected columns
housing_col = 'Total housing units' if 'Total housing units' in df.columns else None
population_col = 'Total Population' if 'Total Population' in df.columns else None

if not housing_col:
    print("Warning: 'Total housing units' column not found in data!")
else:
    print(f"Housing column found: {housing_col}")

if not population_col:
    print("Warning: 'Total Population' column not found in data!")
else:
    print(f"Population column found: {population_col}")

# Missouri coordinates
MO_CENTER_LAT = 38.5767
MO_CENTER_LON = -92.1735

# Create a lookup dictionary for Missouri ZIP codes
mo_zip_coords = {
    # St. Louis area
    '63101': (38.6270, -90.1994),
    '63102': (38.6246, -90.1898),
    '63103': (38.6361, -90.2146),
    '63104': (38.6118, -90.2147),
    '63105': (38.6449, -90.3176),
    '63106': (38.6439, -90.2004),
    '63107': (38.6709, -90.2216),
    '63108': (38.6492, -90.2529),
    '63109': (38.5890, -90.2687),
    '63110': (38.6235, -90.2609),
    
    # Kansas City area
    '64101': (39.0997, -94.5786),
    '64102': (39.1084, -94.5935),
    '64105': (39.1050, -94.5830),
    '64106': (39.1099, -94.5741),
    '64108': (39.0911, -94.5829),
    '64109': (39.0765, -94.5652),
    '64110': (39.0401, -94.5741),
    '64111': (39.0570, -94.5959),
    '64112': (39.0414, -94.5959),
    '64113': (39.0247, -94.5945),
    
    # Springfield
    '65801': (37.2153, -93.2921),
    '65802': (37.2090, -93.2923),
    '65803': (37.2494, -93.2894),
    '65804': (37.1792, -93.2923),
    '65806': (37.2076, -93.3059),
    '65807': (37.1652, -93.3201),
    
    # Columbia
    '65201': (38.9517, -92.3341),
    '65202': (39.0220, -92.3377),
    '65203': (38.9067, -92.3432),
    '65205': (38.9518, -92.3265),
    
    # Jefferson City
    '65101': (38.5767, -92.1735),
    '65102': (38.5733, -92.1735),
    '65109': (38.5893, -92.2471),
    '65110': (38.5345, -92.1819),
    
    # Other major cities
    '63301': (38.7881, -90.4974),  # St. Charles
    '64801': (37.0842, -94.5135),  # Joplin
    '63701': (37.3059, -89.5181),  # Cape Girardeau
    '65401': (37.9484, -91.7715),  # Rolla
    '64601': (39.7948, -93.5527),  # Chillicothe
    '63501': (40.1947, -92.5830),  # Kirksville
    '63857': (36.7766, -89.5695),  # Sikeston
    '63901': (36.7792, -90.4223)   # Poplar Bluff
}

def create_basic_map(column_name):
    """Create a basic map with markers and circles for a column"""
    try:
        # Create a map centered on Missouri
        m = folium.Map(location=[MO_CENTER_LAT, MO_CENTER_LON], zoom_start=7)
        
        # Add points from data
        for _, row in df.iterrows():
            try:
                # Get ZIP code
                zipcode = row['Zipcode']
                
                # Get column value
                value = float(row[column_name])
                
                # Get housing value for tooltip
                housing_value = float(row[housing_col]) if housing_col in row else None
                
                # Find coordinates
                lat, lon = None, None
                if zipcode in mo_zip_coords:
                    lat, lon = mo_zip_coords[zipcode]
                else:
                    # Try to match first 3 digits
                    prefix = zipcode[:3]
                    for z, coords in mo_zip_coords.items():
                        if z.startswith(prefix):
                            lat, lon = coords
                            break
                
                if lat and lon:
                    # Create tooltip
                    tooltip = f"ZIP: {zipcode}, {column_name}: {value:,.0f}"
                    if housing_value:
                        tooltip += f", Housing: {housing_value:,.0f}"
                    
                    # Create popup
                    popup = f"<b>ZIP:</b> {zipcode}<br><b>{column_name}:</b> {value:,.2f}"
                    if housing_value:
                        popup += f"<br><b>Housing Units:</b> {housing_value:,.0f}"
                    
                    # Add marker
                    radius = max(5, min(20, value / 10000))  # Scale radius based on value
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=popup,
                        tooltip=tooltip,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.7
                    ).add_to(m)
            except Exception as e:
                continue  # Skip problematic entries
                
        # Add title
        title_html = f'''
            <h3 align="center" style="font-size:16px">
                <b>Distribution of {column_name} across Missouri ZIP Codes</b>
            </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    except Exception as e:
        print(f"Error creating map for {column_name}: {str(e)}")
        return None

# Function to create a safe filename
def safe_filename(text):
    # Remove any characters that aren't alphanumeric or underscore
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(text).lower())

# Create maps for key columns
columns_to_map = []

if housing_col:
    columns_to_map.append(housing_col)
if population_col:
    columns_to_map.append(population_col)

# Add a few more columns if they exist
other_columns = ['Median age (years)', 'Households Mean income (dollars)', 'Percent with a disability']
for col in other_columns:
    if col in df.columns:
        columns_to_map.append(col)

# If we still have no columns, use the first few numeric columns
if not columns_to_map:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
    columns_to_map = numeric_cols

print(f"Creating maps for: {columns_to_map}")

# Create a map for each column
for column in columns_to_map:
    try:
        print(f"Creating map for {column}...")
        map_obj = create_basic_map(column)
        
        if map_obj:
            # Save the map
            output_file = f"{safe_filename(column)}_map.html"
            output_path = output_dir / output_file
            map_obj.save(str(output_path))
            print(f"Map saved to {output_path}")
    except Exception as e:
        print(f"Error processing {column}: {str(e)}")

print("All maps created successfully.")
