from geopy.geocoders import Nominatim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize the geolocator
geolocator = Nominatim(user_agent="property_predictor")


# Function to get latitude and longitude
def get_lat_lon(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding address {address}: {e}")
        return None, None


def process_location_data(properties_df):
    # Create new columns for latitude and longitude
    properties_df['lat'] = None
    properties_df['lon'] = None

    # Loop over the rows in the DataFrame and get lat/lon for each address
    for idx, row in properties_df.iterrows():
        address = row['address']  # Assuming column is named 'Address'

        # Get latitude and longitude
        lat, lon = get_lat_lon(address)

        # Assign lat/lon values to the corresponding row
        properties_df.at[idx, 'lat'] = lat
        properties_df.at[idx, 'lon'] = lon

    # Normalize latitude and longitude using Min-Max scaling
    lat_lon = properties_df[['lat', 'lon']].values

    # Initialize the MinMaxScaler to scale to a range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit and transform the latitude and longitude values
    normalized_lat_lon = scaler.fit_transform(lat_lon)

    # Add the normalized lat/lon values back to the DataFrame
    properties_df[['lat_normalized', 'lon_normalized']] = normalized_lat_lon

    return properties_df
