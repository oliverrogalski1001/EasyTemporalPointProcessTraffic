import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class GraphGenerator:
    def __init__(self, data_path="../data/traffic_nodes/NodesCrashData.csv") -> None:
        data: pd.DataFrame = pd.read_csv(data_path)
        data["node"] = data["combined_neighborhood"]
        data = data.drop(columns=["combined_neighborhood"])
        data = data[data["node"] != -1]
        self.data = data

    def process_fn(self, dayofweek, hour):
        # Define functions
        def classify_speed_limit(x):
            x = int(x)
            if 0 <= x < 5:
                return 0
            elif 5 <= x < 10:
                return 5
            elif 10 <= x < 15:
                return 10
            elif 15 <= x < 20:
                return 15
            elif 20 <= x < 25:
                return 20
            elif 25 <= x < 30:
                return 25
            elif 30 <= x < 35:
                return 30
            elif 35 <= x < 40:
                return 35
            elif 40 <= x < 45:
                return 40
            elif 45 <= x < 50:
                return 45
            elif 50 <= x < 55:
                return 50
            elif 55 <= x < 60:
                return 55
            elif 60 <= x < 65:
                return 60
            elif 65 <= x < 70:
                return 65
            elif 70 <= x < 75:
                return 70
            else:
                return 0

        def classify_control_device(x):
            x = str(x).strip()
            if x == "NO CONTROLS":
                return 0
            elif x == "TRAFFIC SIGNAL" or x == "FLASHING CONTROL SIGNAL":
                return 1
            elif x == "STOP SIGN/FLASHER":
                return 2
            elif x == "UNKNOWN":
                return 3
            elif x in [
                "OTHER",
                "OTHER REG. SIGN",
                "OTHER WARNING SIGN",
                "NO PASSING",
                "LANE USE MARKING",
                "YIELD",
                "RAILROAD CROSSING GATE",
                "RR CROSSING SIGN",
                "OTHER RAILROAD CROSSING",
                "PEDESTRIAN CROSSING SIGN",
                "BICYCLE CROSSING SIGN",
                "SCHOOL ZONE",
                "DELINEATORS",
                "POLICE/FLAGMAN",
            ]:
                return 4
            else:
                return 3

        def classify_device_condition(x):
            if x == "NO CONTROLS":
                return 0
            elif x == "FUNCTIONING PROPERLY":
                return 1
            elif x == "UNKNOWN" or x == "OTHER":
                return 2
            elif x == "FUNCTIONING IMPROPERLY" or x == "NOT FUNCTIONING":
                return 3
            elif x == "WORN REFLECTIVE MATERIAL":
                return 4
            elif x == "MISSING":
                return 5
            else:
                return 2

        def classify_weather_condition(x):
            if x == "CLEAR":
                return 0
            elif x == "RAIN":
                return 1
            elif x == "UNKNOWN" or x == "OTHER":
                return 2
            elif x == "SNOW":
                return 3
            elif x == "CLOUDY/OVERCAST":
                return 4
            elif x == "FREEZING RAIN/DRIZZLE":
                return 5
            elif x == "FOG/SMOKE/HAZE":
                return 6
            elif x == "SLEET/HAIL":
                return 7
            elif x == "BLOWING SNOW":
                return 8
            elif x == "SEVERE CROSS WIND GATE":
                return 9
            elif x == "BLOWING SAND, SOIL, DIRT":
                return 10
            else:
                return 2

        def classify_lighting_condition(x):
            if x == "DAYLIGHT":
                return 0
            elif x == "DARKNESS, LIGHTED ROAD":
                return 1
            elif x == "UNKNOWN":
                return 2
            elif x == "DARKNESS":
                return 3
            elif x == "DUSK":
                return 4
            elif x == "DAWN":
                return 5
            else:
                return 2

        def classify_first_crash_type(x):
            if x == "PARKED MOTOR VEHICLE":
                return 0
            elif x == "REAR END":
                return 1
            elif x == "SIDESWIPE SAME DIRECTION":
                return 2
            elif x == "TURNING":
                return 3
            elif x == "ANGLE":
                return 4
            elif (
                x == "FIXED OBJECT"
                or x == "OTHER OBJECT"
                or x == "ANIMAL"
                or x == "TRAIN"
            ):
                return 5
            elif x == "PEDESTRIAN":
                return 6
            elif x == "PEDALCYCLIST":
                return 7
            elif x == "REAR TO FRONT":
                return 8
            elif x == "SIDESWIPE OPPOSITE DIRECTION":
                return 9
            elif x == "REAR TO SIDE":
                return 10
            elif x == "HEAD ON":
                return 11
            elif x == "REAR TO REAR":
                return 12
            elif x == "OTHER NONCOLLISION":
                return 13
            elif x == "OVERTURNED":
                return 14
            else:
                return 15

        def classify_trafficway_type(x):
            if x == "NOT DIVIDED":
                return 0
            elif x == "DIVIDED - W/MEDIAN (NOT RAISED)":
                return 1
            elif x == "FOUR WAY" or x == "FIVE POINT, OR MORE":
                return 2
            elif x == "ONE-WAY":
                return 3
            elif (
                x == "PARKING LOT"
                or x == "ALLEY"
                or x == "DRIVEWAY"
                or x == "ROUNDABOUT"
            ):
                return 4
            elif x == "DIVIDED - W/MEDIAN BARRIER":
                return 5
            elif x == "OTHER" or x == "NOT REPORTED" or x == "UNKNOWN":
                return 6
            elif (
                x == "T-INTERSECTION"
                or x == "UNKNOWN INTERSECTION TYPE"
                or x == "Y-INTERSECTION"
                or x == "L-INTERSECTION"
            ):
                return 7
            elif x == "CENTER TURN LANE" or x == "RAMP" or x == "TRAFFIC ROUTE":
                return 8
            else:
                return 6

        def classify_alignment(x):
            x = str(x).strip()
            if x == "STRAIGHT AND LEVEL":
                return 0
            elif (
                x == "CURVE, LEVEL"
                or x == "CURVE ON GRADE"
                or x == "CURVE ON HILLCREST"
            ):
                return 1
            elif x == "STRAIGHT ON HILLCREST" or x == "STRAIGHT ON GRADE":
                return 2
            else:
                return 3

        def classify_surface(x):
            x = str(x).strip()
            if x == "DRY":
                return 0
            elif x in ["WET", "SNOW OR SLUSH", "ICE", "SAND, MUD, DIRT"]:
                return 1
            else:
                return 2

        def classify_defect(x):
            x = str(x).strip()
            if x == "NO DEFECTS":
                return 0
            elif x in [
                "RUT, HOLES",
                "WORN SURFACE",
                "SHOULDER DEFECT",
                "DEBRIS ON ROADWAY",
            ]:
                return 1
            elif x == "OTHER" or x == "UNKNOWN":
                return 2
            else:
                return 2

        def classify_crash_type(x):
            if x == "NO INJURY / DRIVE AWAY":
                return 0
            elif x == "INJURY AND / OR TOW DUE TO CRASH":
                return 1
            else:
                return 2

        def classify_damage(x):
            x = str(x).strip()
            if x == "$500 OR LESS":
                return 0
            elif x == "$501 - $1,500":
                return 1
            elif x == "OVER $1,500":
                return 2
            else:
                return 3

        def classify_cause(x):
            x = str(x).strip()
            if x == "UNABLE TO DETERMINE" or x == "NOT APPLICABLE":
                return 0
            elif x == "FAILING TO YIELD RIGHT-OF-WAY":
                return 1
            elif x == "FOLLOWING TOO CLOSELY":
                return 2
            elif (
                x == "IMPROPER OVERTAKING/PASSING" or x == "PASSING STOPPED SCHOOL BUS"
            ):
                return 3
            elif x == "IMPROPER BACKING":
                return 4
            elif x == "IMPROPER LANE USAGE" or x == "DRIVING ON WRONG SIDE/WRONG WAY":
                return 5
            elif x == "DRIVING SKILLS/KNOWLEDGE/EXPERIENCE":
                return 6
            elif x == "IMPROPER TURNING/NO SIGNAL" or x == "TURNING RIGHT ON RED":
                return 7
            elif x in [
                "DISREGARDING TRAFFIC SIGNALS",
                "DISREGARDING STOP SIGN",
                "DISREGARDING OTHER TRAFFIC SIGNS",
                "DISREGARDING ROAD MARKINGS",
                "DISREGARDING YIELD SIGN",
            ]:
                return 8
            elif x == "WEATHER":
                return 9
            elif (
                x
                == "OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER"
            ):
                return 10
            elif x in [
                "DISTRACTION - FROM INSIDE VEHICLE",
                "DISTRACTION - FROM OUTSIDE VEHICLE",
                "CELL PHONE USE OTHER THAN TEXTING",
                "DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)",
                "TEXTING",
                "VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)",
            ]:
                return 11
            elif x in [
                "UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)",
                "PHYSICAL CONDITION OF DRIVER",
                "HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)",
            ]:
                return 12
            elif x in [
                "EXCEEDING AUTHORIZED SPEED LIMIT",
                "FAILING TO REDUCE SPEED TO AVOID CRASH",
                "EXCEEDING SAFE SPEED FOR CONDITIONS",
                "EQUIPMENT - VEHICLE CONDITION",
            ]:
                return 13
            elif x in [
                "ANIMAL",
                "OBSTRUCTED CROSSWALKS",
                "EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST",
                "BICYCLE ADVANCING LEGALLY ON RED LIGHT",
                "MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT",
                "ROAD ENGINEERING/SURFACE/MARKING DEFECTS",
                "ROAD CONSTRUCTION/MAINTENANCE",
                "RELATED TO BUS STOP",
            ]:
                return 14
            else:
                return 0

        def num_units(x):
            if pd.isna(x):
                return -1
            elif int(x) > 4:
                return 5
            else:
                return int(x)

        def most_severe_injury(x):
            if x == "NO INDICATION OF INJURY":
                return 0
            elif x == "NONINCAPACITATING INJURY":
                return 1
            elif x == "REPORTED, NOT EVIDENT":
                return 2
            elif x == "INCAPACITATING INJURY":
                return 3
            elif x == "FATAL":
                return 4
            else:
                return 5

        df_copy = self.data.copy()
        filtered_df = df_copy[
            (df_copy["CRASH_DAY_OF_WEEK"] == dayofweek)
            & (df_copy["CRASH_HOUR"] == hour)
        ]

        # Apply all functions
        filtered_df.loc[:, "POSTED_SPEED_LIMIT"] = filtered_df[
            "POSTED_SPEED_LIMIT"
        ].apply(classify_speed_limit)
        filtered_df.loc[:, "TRAFFIC_CONTROL_DEVICE"] = filtered_df[
            "TRAFFIC_CONTROL_DEVICE"
        ].apply(classify_control_device)
        filtered_df.loc[:, "DEVICE_CONDITION"] = filtered_df["DEVICE_CONDITION"].apply(
            classify_device_condition
        )
        filtered_df.loc[:, "WEATHER_CONDITION"] = filtered_df[
            "WEATHER_CONDITION"
        ].apply(classify_weather_condition)
        filtered_df.loc[:, "LIGHTING_CONDITION"] = filtered_df[
            "LIGHTING_CONDITION"
        ].apply(classify_lighting_condition)
        filtered_df.loc[:, "FIRST_CRASH_TYPE"] = filtered_df["FIRST_CRASH_TYPE"].apply(
            classify_first_crash_type
        )
        filtered_df.loc[:, "TRAFFICWAY_TYPE"] = filtered_df["TRAFFICWAY_TYPE"].apply(
            classify_trafficway_type
        )
        filtered_df.loc[:, "ALIGNMENT"] = filtered_df["ALIGNMENT"].apply(
            classify_alignment
        )
        filtered_df.loc[:, "ROADWAY_SURFACE_COND"] = filtered_df[
            "ROADWAY_SURFACE_COND"
        ].apply(classify_surface)
        filtered_df.loc[:, "ROAD_DEFECT"] = filtered_df["ROAD_DEFECT"].apply(
            classify_defect
        )
        filtered_df.loc[:, "CRASH_TYPE"] = filtered_df["CRASH_TYPE"].apply(
            classify_crash_type
        )
        filtered_df.loc[:, "DAMAGE"] = filtered_df["DAMAGE"].apply(classify_damage)
        filtered_df.loc[:, "PRIM_CONTRIBUTORY_CAUSE"] = filtered_df[
            "PRIM_CONTRIBUTORY_CAUSE"
        ].apply(classify_cause)
        filtered_df.loc[:, "SEC_CONTRIBUTORY_CAUSE"] = filtered_df[
            "SEC_CONTRIBUTORY_CAUSE"
        ].apply(classify_cause)
        filtered_df.loc[:, "NUM_UNITS"] = filtered_df["NUM_UNITS"].apply(num_units)
        filtered_df.loc[:, "MOST_SEVERE_INJURY"] = filtered_df[
            "MOST_SEVERE_INJURY"
        ].apply(most_severe_injury)

        # Drop unnecessary columns:
        filtered_df = filtered_df.drop(
            columns=[
                "CRASH_RECORD_ID",
                "CRASH_DATE_EST_I",
                "REPORT_TYPE",
                "DATE_POLICE_NOTIFIED",
                "PHOTOS_TAKEN_I",
                "STATEMENTS_TAKEN_I",
                "LANE_CNT",
                "INTERSECTION_RELATED_I",
                "NOT_RIGHT_OF_WAY_I",
                "HIT_AND_RUN_I",
                "INJURIES_INCAPACITATING",
                "INJURIES_NON_INCAPACITATING",
                "INJURIES_REPORTED_NOT_EVIDENT",
                "INJURIES_NO_INDICATION",
                "INJURIES_UNKNOWN",
                "LATITUDE",
                "LONGITUDE",
                "LOCATION",
                "WORK_ZONE_I",
                "WORK_ZONE_TYPE",
                "WORKERS_PRESENT_I",
                "DOORING_I",
                "STREET_NO",
                "STREET_DIRECTION",
                "STREET_NAME",
                "Unnamed: 0.1",
                "Unnamed: 0",
                "geometry",
                "index_right",
                "pri_neigh",
                "sec_neigh",
                "shape_area",
                "shape_len",
                "timediff_start",
                "timediff_hours",
                "CRASH_MONTH",
                "BEAT_OF_OCCURRENCE",
                "CRASH_DATE",
                "CRASH_DAY_OF_WEEK",
                "CRASH_HOUR",
            ]
        )

        # Return dataframe
        return filtered_df

    def aggregate_and_correlate(self, dayofweek, hour):
        # Process the data
        processed_df = self.process_fn(dayofweek, hour)

        # Ensure 'node' column is present in processed_df
        if "node" not in processed_df.columns:
            raise ValueError(
                "The 'node' column is not present in the processed DataFrame."
            )

        # List of features to be included (replace with your actual feature columns)
        feature_columns = [
            "POSTED_SPEED_LIMIT",
            "TRAFFIC_CONTROL_DEVICE",
            "DEVICE_CONDITION",
            "WEATHER_CONDITION",
            "LIGHTING_CONDITION",
            "FIRST_CRASH_TYPE",
            "TRAFFICWAY_TYPE",
            "ALIGNMENT",
            "ROADWAY_SURFACE_COND",
            "ROAD_DEFECT",
            "CRASH_TYPE",
            "DAMAGE",
            "PRIM_CONTRIBUTORY_CAUSE",
            "SEC_CONTRIBUTORY_CAUSE",
            "NUM_UNITS",
            "MOST_SEVERE_INJURY",
            "INJURIES_TOTAL",
            "INJURIES_FATAL",
            "severity_level",
        ]

        # Aggregate the data to get the average of each feature for each node
        node_feature_means = processed_df.groupby("node")[feature_columns].mean()

        # Normalize the features to have mean 0 and standard deviation 1
        normalized_features = (
            node_feature_means - node_feature_means.mean()
        ) / node_feature_means.std()

        # Return the normalized matrix
        return normalized_features

    def compute_node_connectivity_matrix(self, dayofweek, hour, threshold):
        # Generate the 81x18 matrix
        feature_matrix = self.aggregate_and_correlate(dayofweek, hour)

        # Ensure the feature_matrix is not empty
        if feature_matrix.empty:
            print(f"No data available for Day {dayofweek}, Hour {hour}")
            return None

        # Compute the correlation matrix (81x81) for nodes
        correlation_matrix = feature_matrix.T.corr()

        # Apply the threshold to create the connectivity matrix (binary matrix)
        connectivity_matrix = (correlation_matrix > threshold).astype(int)

        pt_adj_matrix = torch.tensor(connectivity_matrix)
        edge_index = torch.where(pt_adj_matrix != 0)
        edge_index = torch.stack(edge_index, dim=0)

        return edge_index

    def compute_node_connectivity_matrix_knn(self, dayofweek, hour, k):
        # Generate the 81x18 matrix
        feature_matrix = self.aggregate_and_correlate(dayofweek, hour)

        # Ensure the feature_matrix is not empty
        if feature_matrix.empty:
            print(f"No data available for Day {dayofweek}, Hour {hour}")
            return None

        # Fit KNN to determine the neighbors
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        knn.fit(feature_matrix)
        distances, indices = knn.kneighbors(feature_matrix)

        # Create the connectivity matrix (81x81)
        connectivity_matrix = np.zeros((len(feature_matrix), len(feature_matrix)))

        # Fill the connectivity matrix based on KNN results
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                connectivity_matrix[i, neighbor] = 1

        pt_adj_matrix = torch.tensor(connectivity_matrix)
        edge_index = torch.where(pt_adj_matrix != 0)
        edge_index = torch.stack(edge_index, dim=0)

        return edge_index
