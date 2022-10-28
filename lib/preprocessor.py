"""
Ideas:
Remove np.nan from train_labels.csv -> also remove swe values of 0.0?
"""

from datetime import datetime
from geopandas import GeoDataFrame
from pandas import DataFrame
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def get_polygon_center(pg):
    xy = pg.exterior.coords.xy
    x, y = xy[0], xy[1]
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    assert x1 <= mid_x <= x2, "Something is off: x"
    assert y1 <= mid_y <= y2, "Something is off: y"
    return mid_x, mid_y


def find_closest_station(location, station_locations):
    _min = np.inf
    _min_idx = None
    location = np.array(location)
    for i in range(len(station_locations)):
        curr_loc = np.array(station_locations[i])
        distance = np.sum((location - curr_loc) ** 2)
        if distance < _min:
            _min = distance
            _min_idx = i
    return _min_idx, _min


class PreProcessor:
    data_path: str
    minmax_scaler = MinMaxScaler()
    cell_scaler = MinMaxScaler()

    station_test: DataFrame
    station_train: DataFrame

    station_metadata: DataFrame
    station_meta_dict = {}
    neighbor_map = {}

    submission_format: DataFrame
    cell_train: DataFrame
    cell_geodata: GeoDataFrame

    def __init__(self, data_path='../data/'):
        self.data_path = data_path

        print('Loading Data Files...')
        self.load_data()

        print('Calculating and Adding Cell Polygon Center...')
        self.cell_geo_add_center()

        print('Build Station Meta Dict...')
        self.build_station_meta_dict()

        print('Done with initial Loading.')

    def load_data(self):
        self.station_test = pd.read_csv(self.data_path + 'ground_measures_test.csv')
        self.station_train = pd.read_csv(self.data_path + 'ground_measures_train.csv')
        self.station_metadata = pd.read_csv(self.data_path + 'ground_measures_metadata.csv')

        self.cell_train = pd.read_csv(self.data_path + 'train_labels.csv')
        self.cell_geodata = gpd.read_file(self.data_path + 'grid_cells.geojson')
        self.submission_format = pd.read_csv(self.data_path + 'submission_format.csv')

    def cell_geo_add_center(self):
        self.cell_geodata['center'] = self.cell_geodata['geometry']
        self.cell_geodata['center'] = self.cell_geodata['center'].apply(get_polygon_center)

    def create_feature(self, elev, lat, long, date):
        # encode time using cos and sin
        enc_year = self.encode_time(date.year - 2013, 22)  # improves year range
        enc_month = self.encode_time(date.month, 12)
        enc_day = self.encode_time(date.day, 31)

        feature = np.array([
            elev,
            lat,
            long,
            enc_year[0],
            enc_year[1],
            enc_month[0],
            enc_month[1],
            enc_day[0],
            enc_day[1],
        ])
        return feature

    def transform_feature(self, feature, fit=False):
        to_transform = feature[:, :3]
        not_to_transform = feature[:, 3:].copy()
        transformed = self.min_max_scaling(to_transform, fit=fit)

        feature[:, :3] = transformed

        assert (feature[:, 3:] == not_to_transform).all()

        return feature

    def station_knn_impute(self):
        print('Imputing Station Train Data with KNN...')
        print('NaNs in station_test: ' + str(self.station_train.isna().sum().sum()))
        data = self.station_train.loc[:, self.station_train.columns != 'Unnamed: 0']
        impute = KNNImputer()
        data = impute.fit_transform(data)
        self.station_train.loc[:, self.station_train.columns != 'Unnamed: 0'] = data
        print('New NaN Count in station_test: ' + str(self.station_train.isna().sum().sum()))

        print('Imputing Station Test Data with KNN...')
        print('NaNs in station_test: ' + str(self.station_test.isna().sum().sum()))
        data = self.station_test.loc[:, self.station_test.columns != 'Unnamed: 0']
        impute = KNNImputer()
        data = impute.fit_transform(data)
        self.station_test.loc[:, self.station_test.columns != 'Unnamed: 0'] = data
        print('New NaN Count in station_test: ' + str(self.station_test.isna().sum().sum()))

    def build_station_meta_dict(self):
        for oi, r in self.station_metadata.iterrows():
            self.station_meta_dict[r[0]] = {
                'elev': r['elevation_m'],
                'lat': r['latitude'],
                'long': r['longitude']
            }

    def min_max_scaling(self, x, fit=False):
        if fit:
            self.minmax_scaler = self.minmax_scaler.fit(x)
        return self.minmax_scaler.transform(x)

    def encode_time(self, x, max_val):
        sin_x = np.sin(2 * np.pi * x / max_val)
        cos_x = np.cos(2 * np.pi * x / max_val)
        return sin_x, cos_x

    def build_x_y_vectors(self, dataframe):
        x = []
        y = []
        for oi, j in dataframe.iterrows():
            station = self.station_meta_dict[j[0]]
            for k, e in j.items():
                if k == 'Unnamed: 0':
                    continue
                dt = datetime.strptime(k, '%Y-%m-%d')
                date = dt.date()

                feature = self.create_feature(station['elev'], station['lat'], station['long'], date)
                x.append(feature)
                y.append(e)
        return np.array(x), np.array(y)

    def get_labels_x_y(self):
        """
        This function returns only the rows from train_labels.csv which are present in submission.csv
        organized in X and y
        """
        train_labels = self.cell_train[self.cell_train["cell_id"].isin(self.submission_format["cell_id"])].reset_index(
            drop=True)
        ground_measures = self.station_metadata
        ground_measures["coord"] = np.empty((len(ground_measures), 0)).tolist()
        for i, row in ground_measures.iterrows():
            ground_measures.at[i, "coord"] = (row[4], row[3])
        ids_train = train_labels["cell_id"]
        ids_geo = self.cell_geodata["cell_id"]
        train_labels["location"] = self.cell_geodata[ids_geo.isin(ids_train)].reset_index()["geometry"]
        train_labels["location"] = train_labels["location"].apply(get_polygon_center)
        train_labels["elev"] = np.nan
        for i, row in train_labels.iterrows():
            location = row[-2]
            station_locations = ground_measures["coord"]
            idx, _ = find_closest_station(location, station_locations)
            closest_elev = ground_measures.at[idx, "elevation_m"]
            train_labels.at[i, "elev"] = closest_elev
        x_labels = []
        y_labels = []
        dates = list(train_labels.columns)[1:-2]
        for i, row in train_labels.iterrows():
            elev = row[-1]
            location = row[-2]
            for d, swe in zip(dates, row[1:-2]):
                date = datetime.strptime(d, '%Y-%m-%d')

                feature = self.create_feature(elev, location[1], location[0], date)

                if not np.isnan(swe):
                    x_labels.append(feature)
                    y_labels.append(swe)
        return np.array(x_labels), np.array(y_labels)

    def get_dataset(self):
        x_train, y_train = self.build_x_y_vectors(self.station_train)
        x_test, y_test = self.build_x_y_vectors(self.station_test)
        x_labels, y_labels = self.get_labels_x_y()
        x = np.vstack((x_train, x_test, x_labels))
        y = np.vstack((y_train[:, None], y_test[:, None], y_labels[:, None]))

        return self.transform_feature(x, fit=True), y

    def build_station_neighbor_map(self, n=3):
        print('Building Neighbor Dict...')

        for i, r in self.cell_geodata.iterrows():
            center = self.cell_geodata.loc[self.cell_geodata['cell_id'] == r['cell_id'], 'center'].values[0]
            lat = center[1]
            long = center[0]
            filter_distance = 0.5
            filtered_stations = self.station_metadata[(self.station_metadata['longitude'] >= long - filter_distance) & (
                    self.station_metadata['longitude'] <= long + filter_distance)]
            filtered_stations = filtered_stations[(filtered_stations['latitude'] >= lat - filter_distance) & (
                    filtered_stations['latitude'] <= lat + filter_distance)]

            def get_neighbor_dict(stations):
                neighbors = {}
                for ii, s in stations.iterrows():
                    diff_long = abs(s['longitude'] - long)
                    diff_lat = abs(s['latitude'] - lat)
                    neighbors[s[0]] = [diff_lat + diff_long, diff_long, diff_lat]
                return neighbors

            neighbor_dict = get_neighbor_dict(filtered_stations)
            if len(neighbor_dict) < 3:
                neighbor_dict = get_neighbor_dict(self.station_metadata)

            sorted_neighbors = sorted(neighbor_dict.items(), key=lambda sx: abs(sx[1][0]))
            self.neighbor_map[r['cell_id']] = [[x[0], x[1][0], x[1][1], x[1][2]] for x in sorted_neighbors[:n]]
            print('%s/%s' % (i + 1, len(self.cell_train)), end='\r')
        print('Done................')

    def create_submission(self, filename, model):
        """
        Creates submission for given model.
        :param filename: filename for saving
        :param model: model for prediction
        :return:
        """
        geodata = gpd.read_file('data/grid_cells.geojson')
        ground_measures = pd.read_csv("data/ground_measures_metadata.csv")
        submission = pd.read_csv("data/submission_format.csv")
        submission["location"] = np.nan
        submission["elev"] = np.nan
        ids_geo = geodata["cell_id"]
        ids_sub = submission["cell_id"]
        ground_measures["coord"] = np.empty((len(ground_measures), 0)).tolist()
        submission["location"] = geodata[ids_geo.isin(ids_sub)].reset_index()["geometry"]
        submission["location"] = submission["location"].apply(get_polygon_center)
        for i, row in ground_measures.iterrows():
            ground_measures.at[i, "coord"] = (row[4], row[3])
        for i, row in submission.iterrows():
            location = row[-2]
            station_locations = ground_measures["coord"]
            idx, _ = find_closest_station(location, station_locations)
            closest_elev = ground_measures.at[idx, "elevation_m"]
            submission.at[i, "elev"] = closest_elev
        dates = list(submission.columns)[1:-2]
        with tqdm(range(len(submission))) as pbar:
            for _, (i, row) in zip(pbar, submission.iterrows()):
                elev = row[-1]
                location = row[-2]
                batched = []
                for d in dates:
                    date = datetime.strptime(d, '%Y-%m-%d')
                    feature = self.create_feature(elev, location[1], location[0], date).reshape(1, -1)
                    batched.append(self.transform_feature(feature))

                prediction = model.predict(np.array(batched))
                for pred, d in zip(prediction, dates):
                    submission.at[i, d] = pred

        submission.drop(["elev", "location"], axis=1).to_csv(filename, index=False)


if __name__ == '__main__':
    pp = PreProcessor(data_path='../data/')
    pp.station_knn_impute()
    x, y = pp.get_dataset()
