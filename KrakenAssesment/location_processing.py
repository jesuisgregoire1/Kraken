from geopy.geocoders import Nominatim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class LocationProcessor:
    def __init__(self, user_agent="property_predictor"):
        # initialize the geolocator with a user-agent string
        self.geolocator = Nominatim(user_agent=user_agent)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # initialize the MinMaxScaler

    def get_lat_lon(self, address):
        try:
            location = self.geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except Exception as e:
            print(f"Error geocoding address {address}: {e}")
            return None, None

    def process_location_data(self, properties_df):
        # create new columns for latitude and longitude
        properties_df['lat'] = None
        properties_df['lon'] = None
        # loop over the rows in the DataFrame
        for idx, row in properties_df.iterrows():
            address = row['address']  # assuming column is named 'address'

            # get latitude and longitude
            lat, lon = self.get_lat_lon(address)

            # assign lat/lon values to the corresponding row
            properties_df.at[idx, 'lat'] = lat
            properties_df.at[idx, 'lon'] = lon

        # normalize latitude and longitude using Min-Max scaling
        lat_lon = properties_df[['lat', 'lon']].values

        # fit and transform the latitude and longitude values
        normalized_lat_lon = self.scaler.fit_transform(lat_lon)
        # add the normalized lat/lon values back to the DataFrame
        properties_df[['lat_normalized', 'lon_normalized']] = normalized_lat_lon

        return properties_df
