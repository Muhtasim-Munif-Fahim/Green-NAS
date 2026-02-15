"""
Global Weather Data Downloader
Downloads hourly weather data for 200 global cities using Open-Meteo API

Free API, no authentication required, 80 years of historical data!
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from tqdm import tqdm


# 200 Global Cities with Coordinates
GLOBAL_CITIES = {
    # ASIA (75 cities)
    ## South Asia (10)
    'Dhaka': {'lat': 23.8103, 'lon': 90.4125, 'country': 'Bangladesh', 'tier': 1},
    'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'country': 'India', 'tier': 1},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'country': 'India', 'tier': 1},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'country': 'India', 'tier': 1},
    'Lahore': {'lat': 31.5497, 'lon': 74.3436, 'country': 'Pakistan', 'tier': 1},
    'Karachi': {'lat': 24.8607, 'lon': 67.0011, 'country': 'Pakistan', 'tier': 1},
    'Colombo': {'lat': 6.9271, 'lon': 79.8612, 'country': 'Sri Lanka', 'tier': 1},
    'Kathmandu': {'lat': 27.7172, 'lon': 85.3240, 'country': 'Nepal', 'tier': 1},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'country': 'India', 'tier': 2},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'country': 'India', 'tier': 2},
    
    ## Southeast Asia (15)
    'Bangkok': {'lat': 13.7563, 'lon': 100.5018, 'country': 'Thailand', 'tier': 1},
    'Hanoi': {'lat': 21.0285, 'lon': 105.8542, 'country': 'Vietnam', 'tier': 1},
    'Jakarta': {'lat': -6.2088, 'lon': 106.8456, 'country': 'Indonesia', 'tier': 1},
    'Manila': {'lat': 14.5995, 'lon': 120.9842, 'country': 'Philippines', 'tier': 1},
    'Singapore': {'lat': 1.3521, 'lon': 103.8198, 'country': 'Singapore', 'tier': 1},
    'Kuala Lumpur': {'lat': 3.1390, 'lon': 101.6869, 'country': 'Malaysia', 'tier': 1},
    'Ho Chi Minh': {'lat': 10.8231, 'lon': 106.6297, 'country': 'Vietnam', 'tier': 2},
    'Yangon': {'lat': 16.8661, 'lon': 96.1951, 'country': 'Myanmar', 'tier': 1},
    'Phnom Penh': {'lat': 11.5564, 'lon': 104.9282, 'country': 'Cambodia', 'tier': 2},
    'Vientiane': {'lat': 17.9757, 'lon': 102.6331, 'country': 'Laos', 'tier': 2},
    'Surabaya': {'lat': -7.2575, 'lon': 112.7521, 'country': 'Indonesia', 'tier': 2},
    'Cebu': {'lat': 10.3157, 'lon': 123.8854, 'country': 'Philippines', 'tier': 3},
    'Chiang Mai': {'lat': 18.7883, 'lon': 98.9853, 'country': 'Thailand', 'tier': 2},
    'Bandung': {'lat': -6.9175, 'lon': 107.6191, 'country': 'Indonesia', 'tier': 2},
    'George Town': {'lat': 5.4141, 'lon': 100.3288, 'country': 'Malaysia', 'tier': 3},
    
    ## East Asia (15)
    'Beijing': {'lat': 39.9042, 'lon': 116.4074, 'country': 'China', 'tier': 1},
    'Shanghai': {'lat': 31.2304, 'lon': 121.4737, 'country': 'China', 'tier': 1},
    'Tokyo': {'lat': 35.6762, 'lon': 139.6503, 'country': 'Japan', 'tier': 1},
    'Seoul': {'lat': 37.5665, 'lon': 126.9780, 'country': 'South Korea', 'tier': 1},
    'Hong Kong': {'lat': 22.3193, 'lon': 114.1694, 'country': 'Hong Kong', 'tier': 1},
    'Taipei': {'lat': 25.0330, 'lon': 121.5654, 'country': 'Taiwan', 'tier': 1},
    'Osaka': {'lat': 34.6937, 'lon': 135.5023, 'country': 'Japan', 'tier': 1},
    'Guangzhou': {'lat': 23.1291, 'lon': 113.2644, 'country': 'China', 'tier': 2},
    'Shenzhen': {'lat': 22.5431, 'lon': 114.0579, 'country': 'China', 'tier': 2},
    'Busan': {'lat': 35.1796, 'lon': 129.0756, 'country': 'South Korea', 'tier': 3},
    'Chongqing': {'lat': 29.4316, 'lon': 106.9123, 'country': 'China', 'tier': 2},
    'Tianjin': {'lat': 39.3434, 'lon': 117.3616, 'country': 'China', 'tier': 2},
    'Wuhan': {'lat': 30.5928, 'lon': 114.3055, 'country': 'China', 'tier': 3},
    'Chengdu': {'lat': 30.5728, 'lon': 104.0668, 'country': 'China', 'tier': 3},
    'Sapporo': {'lat': 43.0642, 'lon': 141.3469, 'country': 'Japan', 'tier': 3},
    
    ## West/Central Asia (10)
    'Dubai': {'lat': 25.2048, 'lon': 55.2708, 'country': 'UAE', 'tier': 1},
    'Istanbul': {'lat': 41.0082, 'lon': 28.9784, 'country': 'Turkey', 'tier': 1},
    'Tehran': {'lat': 35.6892, 'lon': 51.3890, 'country': 'Iran', 'tier': 2},
    'Baghdad': {'lat': 33.3128, 'lon': 44.3615, 'country': 'Iraq', 'tier': 2},
    'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'country': 'Saudi Arabia', 'tier': 3},
    'Doha': {'lat': 25.2854, 'lon': 51.5310, 'country': 'Qatar', 'tier': 3},
    'Kuwait City': {'lat': 29.3759, 'lon': 47.9774, 'country': 'Kuwait', 'tier': 3},
    'Kabul': {'lat': 34.5553, 'lon': 69.2075, 'country': 'Afghanistan', 'tier': 3},
    'Tashkent': {'lat': 41.2995, 'lon': 69.2401, 'country': 'Uzbekistan', 'tier': 3},
    'Almaty': {'lat': 43.2220, 'lon': 76.8512, 'country': 'Kazakhstan', 'tier': 3},
    
    # EUROPE (45 cities)
    ## Western Europe (15)
    'London': {'lat': 51.5074, 'lon': -0.1278, 'country': 'UK', 'tier': 1},
    'Paris': {'lat': 48.8566, 'lon': 2.3522, 'country': 'France', 'tier': 1},
    'Berlin': {'lat': 52.5200, 'lon': 13.4050, 'country': 'Germany', 'tier': 1},
    'Madrid': {'lat': 40.4168, 'lon': -3.7038, 'country': 'Spain', 'tier': 1},
    'Rome': {'lat': 41.9028, 'lon': 12.4964, 'country': 'Italy', 'tier': 1},
    'Amsterdam': {'lat': 52.3676, 'lon': 4.9041, 'country': 'Netherlands', 'tier': 1},
    'Brussels': {'lat': 50.8503, 'lon': 4.3517, 'country': 'Belgium', 'tier': 1},
    'Vienna': {'lat': 48.2082, 'lon': 16.3738, 'country': 'Austria', 'tier': 1},
    'Copenhagen': {'lat': 55.6761, 'lon': 12.5683, 'country': 'Denmark', 'tier': 2},
    'Stockholm': {'lat': 59.3293, 'lon': 18.0686, 'country': 'Sweden', 'tier': 2},
    'Oslo': {'lat': 59.9139, 'lon': 10.7522, 'country': 'Norway', 'tier': 2},
    'Helsinki': {'lat': 60.1699, 'lon': 24.9384, 'country': 'Finland', 'tier': 2},
    'Zurich': {'lat': 47.3769, 'lon': 8.5417, 'country': 'Switzerland', 'tier': 2},
    'Munich': {'lat': 48.1351, 'lon': 11.5820, 'country': 'Germany', 'tier': 3},
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734, 'country': 'Spain', 'tier': 1},
    
    ## Eastern Europe (15)
    'Warsaw': {'lat': 52.2297, 'lon': 21.0122, 'country': 'Poland', 'tier': 1},
    'Moscow': {'lat': 55.7558, 'lon': 37.6173, 'country': 'Russia', 'tier': 1},
    'Prague': {'lat': 50.0755, 'lon': 14.4378, 'country': 'Czech Republic', 'tier': 1},
    'Budapest': {'lat': 47.4979, 'lon': 19.0402, 'country': 'Hungary', 'tier': 1},
    'Bucharest': {'lat': 44.4268, 'lon': 26.1025, 'country': 'Romania', 'tier': 2},
    'Kiev': {'lat': 50.4501, 'lon': 30.5234, 'country': 'Ukraine', 'tier': 2},
    'Sofia': {'lat': 42.6977, 'lon': 23.3219, 'country': 'Bulgaria', 'tier': 2},
    'Belgrade': {'lat': 44.7866, 'lon': 20.4489, 'country': 'Serbia', 'tier': 3},
    'Zagreb': {'lat': 45.8150, 'lon': 15.9819, 'country': 'Croatia', 'tier': 2},
    'Athens': {'lat': 37.9838, 'lon': 23.7275, 'country': 'Greece', 'tier': 1},
    'Lisbon': {'lat': 38.7223, 'lon': -9.1393, 'country': 'Portugal', 'tier': 1},
    'St Petersburg': {'lat': 59.9311, 'lon': 30.3609, 'country': 'Russia', 'tier': 3},
    'Minsk': {'lat': 53.9006, 'lon': 27.5590, 'country': 'Belarus', 'tier': 3},
    'Tallinn': {'lat': 59.4370, 'lon': 24.7536, 'country': 'Estonia', 'tier': 3},
    'Riga': {'lat': 56.9496, 'lon': 24.1052, 'country': 'Latvia', 'tier': 3},
    
    # NORTH AMERICA (25 cities)
    'New York': {'lat': 40.7128, 'lon': -74.0060, 'country': 'USA', 'tier': 1},
    'Los Angeles': {'lat': 34.0522, 'lon': -118.2437, 'country': 'USA', 'tier': 1},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'country': 'USA', 'tier': 1},
    'Houston': {'lat': 29.7604, 'lon': -95.3698, 'country': 'USA', 'tier': 1},
    'Toronto': {'lat': 43.6532, 'lon': -79.3832, 'country': 'Canada', 'tier': 1},
    'Vancouver': {'lat': 49.2827, 'lon': -123.1207, 'country': 'Canada', 'tier': 1},
    'Mexico City': {'lat': 19.4326, 'lon': -99.1332, 'country': 'Mexico', 'tier': 1},
    'Montreal': {'lat': 45.5017, 'lon': -73.5673, 'country': 'Canada', 'tier': 2},
    'Phoenix': {'lat': 33.4484, 'lon': -112.0740, 'country': 'USA', 'tier': 3},
    'Philadelphia': {'lat': 39.9526, 'lon': -75.1652, 'country': 'USA', 'tier': 3},
    'San Francisco': {'lat': 37.7749, 'lon': -122.4194, 'country': 'USA', 'tier': 2},
    'Seattle': {'lat': 47.6062, 'lon': -122.3321, 'country': 'USA', 'tier': 2},
    'Miami': {'lat': 25.7617, 'lon': -80.1918, 'country': 'USA', 'tier': 3},
    'Boston': {'lat': 42.3601, 'lon': -71.0589, 'country': 'USA', 'tier': 3},
    'Dallas': {'lat': 32.7767, 'lon': -96.7970, 'country': 'USA', 'tier': 3},
    'Denver': {'lat': 39.7392, 'lon': -104.9903, 'country': 'USA', 'tier': 3},
    'Guadalajara': {'lat': 20.6597, 'lon': -103.3496, 'country': 'Mexico', 'tier': 1},
    'Monterrey': {'lat': 25.6866, 'lon': -100.3161, 'country': 'Mexico', 'tier': 1},
    'Havana': {'lat': 23.1136, 'lon': -82.3666, 'country': 'Cuba', 'tier': 1},
    'San Jose CR': {'lat': 9.9281, 'lon': -84.0907, 'country': 'Costa Rica', 'tier': 2},
    'Panama City': {'lat': 8.9824, 'lon': -79.5199, 'country': 'Panama', 'tier': 2},
    'Calgary': {'lat': 51.0447, 'lon': -114.0719, 'country': 'Canada', 'tier': 3},
    'Ottawa': {'lat': 45.4215, 'lon': -75.6972, 'country': 'Canada', 'tier': 3},
    'Guatemala City': {'lat': 14.6349, 'lon': -90.5069, 'country': 'Guatemala', 'tier': 3},
    'San Salvador': {'lat': 13.6929, 'lon': -89.2182, 'country': 'El Salvador', 'tier': 3},
    
    # SOUTH AMERICA (19 cities)
    'São Paulo': {'lat': -23.5505, 'lon': -46.6333, 'country': 'Brazil', 'tier': 1},
    'Buenos Aires': {'lat': -34.6037, 'lon': -58.3816, 'country': 'Argentina', 'tier': 1},
    'Lima': {'lat': -12.0464, 'lon': -77.0428, 'country': 'Peru', 'tier': 1},
    'Santiago': {'lat': -33.4489, 'lon': -70.6693, 'country': 'Chile', 'tier': 1},
    'Bogotá': {'lat': 4.7110, 'lon': -74.0721, 'country': 'Colombia', 'tier': 2},
    'Rio de Janeiro': {'lat': -22.9068, 'lon': -43.1729, 'country': 'Brazil', 'tier': 3},
    'Brasília': {'lat': -15.8267, 'lon': -47.9218, 'country': 'Brazil', 'tier': 2},
    'Caracas': {'lat': 10.4806, 'lon': -66.9036, 'country': 'Venezuela', 'tier': 3},
    'Quito': {'lat': -0.1807, 'lon': -78.4678, 'country': 'Ecuador', 'tier': 2},
    'Montevideo': {'lat': -34.9011, 'lon': -56.1645, 'country': 'Uruguay', 'tier': 3},
    'Asunción': {'lat': -25.2637, 'lon': -57.5759, 'country': 'Paraguay', 'tier': 3},
    'La Paz': {'lat': -16.5000, 'lon': -68.1500, 'country': 'Bolivia', 'tier': 3},
    'Medellín': {'lat': 6.2476, 'lon': -75.5658, 'country': 'Colombia', 'tier': 3},
    'Belo Horizonte': {'lat': -19.9167, 'lon': -43.9345, 'country': 'Brazil', 'tier': 3},
    'Porto Alegre': {'lat': -30.0346, 'lon': -51.2177, 'country': 'Brazil', 'tier': 3},
    'Recife': {'lat': -8.0476, 'lon': -34.9870, 'country': 'Brazil', 'tier': 3},
    'Curitiba': {'lat': -25.4284, 'lon': -49.2733, 'country': 'Brazil', 'tier': 3},
    'Guayaquil': {'lat': -2.1709, 'lon': -79.9224, 'country': 'Ecuador', 'tier': 3},
    'Cali': {'lat': 3.4516, 'lon': -76.5320, 'country': 'Colombia', 'tier': 3},
    
    # AFRICA (31 cities)
    'Cairo': {'lat': 30.0444, 'lon': 31.2357, 'country': 'Egypt', 'tier': 1},
    'Lagos': {'lat': 6.5244, 'lon': 3.3792, 'country': 'Nigeria', 'tier': 1},
    'Johannesburg': {'lat': -26.2041, 'lon': 28.0473, 'country': 'South Africa', 'tier': 1},
    'Nairobi': {'lat': -1.2864, 'lon': 36.8172, 'country': 'Kenya', 'tier': 1},
    'Casablanca': {'lat': 33.5731, 'lon': -7.5898, 'country': 'Morocco', 'tier': 1},
    'Accra': {'lat': 5.6037, 'lon': -0.1870, 'country': 'Ghana', 'tier': 1},
    'Addis Ababa': {'lat': 9.0320, 'lon': 38.7469, 'country': 'Ethiopia', 'tier': 2},
    'Dar es Salaam': {'lat': -6.7924, 'lon': 39.2083, 'country': 'Tanzania', 'tier': 2},
    'Kigali': {'lat': -1.9403, 'lon': 29.8739, 'country': 'Rwanda', 'tier': 2},
    'Kampala': {'lat': 0.3476, 'lon': 32.5825, 'country': 'Uganda', 'tier': 2},
    'Tunis': {'lat': 36.8065, 'lon': 10.1815, 'country': 'Tunisia', 'tier': 2},
    'Algiers': {'lat': 36.7538, 'lon': 3.0588, 'country': 'Algeria', 'tier': 3},
    'Dakar': {'lat': 14.7167, 'lon': -17.4677, 'country': 'Senegal', 'tier': 3},
    'Abidjan': {'lat': 5.3600, 'lon': -4.0083, 'country': 'Ivory Coast', 'tier': 3},
    'Khartoum': {'lat': 15.5007, 'lon': 32.5599, 'country': 'Sudan', 'tier': 3},
    'Luanda': {'lat': -8.8147, 'lon': 13.2302, 'country': 'Angola', 'tier': 3},
    'Maputo': {'lat': -25.9655, 'lon': 32.5832, 'country': 'Mozambique', 'tier': 3},
    'Harare': {'lat': -17.8252, 'lon': 31.0335, 'country': 'Zimbabwe', 'tier': 3},
    'Lusaka': {'lat': -15.3875, 'lon': 28.3228, 'country': 'Zambia', 'tier': 3},
    'Windhoek': {'lat': -22.5597, 'lon': 17.0832, 'country': 'Namibia', 'tier': 3},
    'Cape Town': {'lat': -33.9249, 'lon': 18.4241, 'country': 'South Africa', 'tier': 3},
    'Tripoli': {'lat': 32.8872, 'lon': 13.1913, 'country': 'Libya', 'tier': 3},
    'Bamako': {'lat': 12.6392, 'lon': -8.0029, 'country': 'Mali', 'tier': 3},
    'Niamey': {'lat': 13.5127, 'lon': 2.1128, 'country': 'Niger', 'tier': 3},
    'Ouagadougou': {'lat': 12.3714, 'lon': -1.5197, 'country': 'Burkina Faso', 'tier': 3},
    'Kinshasa': {'lat': -4.4419, 'lon': 15.2663, 'country': 'DR Congo', 'tier': 3},
    'Yaoundé': {'lat': 3.8480, 'lon': 11.5021, 'country': 'Cameroon', 'tier': 3},
    'Freetown': {'lat': 8.4657, 'lon': -13.2317, 'country': 'Sierra Leone', 'tier': 3},
    'Monrovia': {'lat': 6.3156, 'lon': -10.8074, 'country': 'Liberia', 'tier': 3},
    'Conakry': {'lat': 9.6412, 'lon': -13.5784, 'country': 'Guinea', 'tier': 3},
    'N Djamena': {'lat': 12.1348, 'lon': 15.0557, 'country': 'Chad', 'tier': 3},
    
    # OCEANIA (7 cities)
    'Sydney': {'lat': -33.8688, 'lon': 151.2093, 'country': 'Australia', 'tier': 1},
    'Melbourne': {'lat': -37.8136, 'lon': 144.9631, 'country': 'Australia', 'tier': 2},
    'Auckland': {'lat': -36.8485, 'lon': 174.7633, 'country': 'New Zealand', 'tier': 2},
    'Brisbane': {'lat': -27.4698, 'lon': 153.0251, 'country': 'Australia', 'tier': 3},
    'Perth': {'lat': -31.9505, 'lon': 115.8605, 'country': 'Australia', 'tier': 3},
    'Adelaide': {'lat': -34.9285, 'lon': 138.6007, 'country': 'Australia', 'tier': 3},
    'Wellington': {'lat': -41.2865, 'lon': 174.7762, 'country': 'New Zealand', 'tier': 3},
}


def download_city_weather(city_name, city_info, start_date="2019-01-01", end_date="2024-12-31", 
                          output_dir="data/raw/weather"):
    """
    Download hourly weather data for a city using Open-Meteo API
    
    Args:
        city_name: Name of the city
        city_info: Dict with lat, lon, country, tier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
    
    Returns:
        DataFrame with hourly weather data or None if error
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open-Meteo API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters - request all useful weather variables
    params = {
        "latitude": city_info['lat'],
        "longitude": city_info['lon'],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",          # Temperature at 2m
            "relative_humidity_2m",    # Humidity
            "precipitation",           # Precipitation
            "pressure_msl",            # Sea level pressure
            "surface_pressure",        # Surface pressure
            "cloud_cover",             # Cloud cover %
            "wind_speed_10m",          # Wind speed at 10m
            "wind_direction_10m",      # Wind direction
            "shortwave_radiation",     # Solar radiation
        ],
        "timezone": "UTC"
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        if 'hourly' in data:
            df = pd.DataFrame(data['hourly'])
            
            # Add metadata
            df['city'] = city_name
            df['country'] = city_info['country']
            df['latitude'] = city_info['lat']
            df['longitude'] = city_info['lon']
            df['tier'] = city_info['tier']
            
            # Save to CSV
            filename = f"{city_name.lower().replace(' ', '_')}_{city_info['country'].replace(' ', '_').lower()}_tier{city_info['tier']}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            
            return df
        else:
            print(f"  ❌ No data returned for {city_name}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error downloading {city_name}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error processing {city_name}: {e}")
        return None


def download_all_cities(tier=None, max_cities=None):
    """
    Download weather data for all cities (or filtered selection)
    
    Args:
        tier: If specified, only download cities of this tier (1, 2, or 3)
        max_cities: If specified, limit total downloads
    """
    print("=" * 80)
    print("GLOBAL WEATHER DATA DOWNLOADER")
    print("=" * 80)
    print(f"Source: Open-Meteo Historical Weather API")
    print(f"Period: 2019-2024 (5 years)")
    print(f"Resolution: Hourly")
    print(f"Total cities available: {len(GLOBAL_CITIES)}\n")
    
    # Filter cities if tier specified
    if tier:
        cities_to_download = {k: v for k, v in GLOBAL_CITIES.items() if v['tier'] == tier}
        print(f"Filtering to Tier {tier} cities: {len(cities_to_download)} cities")
    else:
        cities_to_download = GLOBAL_CITIES
    
    # Limit if max_cities specified
    if max_cities:
        cities_to_download = dict(list(cities_to_download.items())[:max_cities])
        print(f"Limiting to first {max_cities} cities")
    
    print(f"\nDownloading {len(cities_to_download)} cities...")
    print("-" * 80)
    
    # Download statistics
    successful = []
    failed = []
    
    # Download each city
    for city_name, city_info in tqdm(cities_to_download.items(), desc="Downloading"):
        print(f"\n{city_name:20s} ({city_info['country']:15s})", end=' ')
        
        df = download_city_weather(city_name, city_info)
        
        if df is not None:
            records = len(df)
            print(f"✅ {records:,} hourly records")
            successful.append({
                'city': city_name,
                'country': city_info['country'],
                'tier': city_info['tier'],
                'records': records
            })
        else:
            failed.append(city_name)
        
        # Rate limiting - be nice to free API
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total cities attempted: {len(cities_to_download)}")
    print(f"Successfully downloaded: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)/len(cities_to_download)*100:.1f}%")
    
    if successful:
        # Save manifest
        manifest = {
            'download_date': datetime.now().isoformat(),
            'source': 'Open-Meteo Historical Weather API',
            'period': '2019-2024',
            'resolution': 'hourly',
            'total_cities': len(successful),
            'cities': successful
        }
        
        manifest_file = 'data/metadata/weather_download_manifest.json'
        os.makedirs('data/metadata', exist_ok=True)
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n💾 Manifest saved to: {manifest_file}")
        
        # Statistics
        total_records = sum(c['records'] for c in successful)
        print(f"\n📊 Data Statistics:")
        print(f"   Total hourly records: {total_records:,}")
        print(f"   Average per city: {total_records/len(successful):,.0f}")
        print(f"   Estimated total size: ~{total_records * 0.0001:.1f} MB")
    
    if failed:
        print(f"\n⚠️  Failed cities: {', '.join(failed)}")
    
    return successful, failed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download global weather data from Open-Meteo")
    parser.add_argument('--tier', type=int, choices=[1, 2, 3], 
                       help='Download only specific tier (1, 2, or 3)')
    parser.add_argument('--max-cities', type=int,
                       help='Limit total number of cities to download')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: download only first 5 cities')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST MODE: Downloading first 5 cities only\n")
        download_all_cities(max_cities=5)
    else:
        download_all_cities(tier=args.tier, max_cities=args.max_cities)


if __name__ == "__main__":
    main()
