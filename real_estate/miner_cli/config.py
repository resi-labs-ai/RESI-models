"""Miner CLI configuration defaults."""

# Network to subnet UID mapping
NETWORK_NETUIDS = {
    "finney": 46,
    "mainnet": 46,
    "test": 428,
    "testnet": 428,
}

# Model validation
MAX_MODEL_SIZE_MB = 200
REQUIRED_ONNX_VERSION = "1.20.0"
REQUIRED_ONNXRUNTIME_VERSION = "1.20.1"

# Chain scanning
SCAN_MAX_BLOCKS = 25
SCAN_MAX_EXTRINSICS_PER_BLOCK = 1000

# Commitment constraints
MAX_REPO_BYTES = 51

# Model interface expectations
EXPECTED_NUM_FEATURES = 72

# Test sample data for local evaluation (from validation set)
# Features exclude 'price' - 72 features total
TEST_FEATURE_NAMES = [
    "living_area_sqft",
    "lot_size_sqft",
    "bedrooms",
    "bathrooms",
    "latitude",
    "longitude",
    "year_built",
    "property_age",
    "days_on_zillow",
    "stories",
    "has_basement",
    "has_garage",
    "has_attached_garage",
    "has_cooling",
    "has_heating",
    "has_fireplace",
    "has_spa",
    "has_view",
    "has_pool",
    "has_open_parking",
    "has_home_warranty",
    "is_new_construction",
    "is_senior_community",
    "has_waterfront_view",
    "bathrooms_full",
    "bathrooms_half",
    "bathrooms_three_quarter",
    "garage_capacity",
    "covered_parking_capacity",
    "open_parking_capacity",
    "parking_capacity_total",
    "fireplaces_count",
    "cooling_count",
    "heating_count",
    "appliances_count",
    "flooring_count",
    "construction_materials_count",
    "interior_features_count",
    "exterior_features_count",
    "community_features_count",
    "parking_features_count",
    "pool_features_count",
    "laundry_features_count",
    "lot_features_count",
    "view_features_count",
    "sewer_count",
    "water_source_count",
    "electric_count",
    "school_count",
    "min_school_distance",
    "avg_school_rating",
    "max_school_rating",
    "elementary_rating",
    "middle_rating",
    "high_rating",
    "has_central_air",
    "has_forced_air_heating",
    "has_natural_gas",
    "has_hardwood_floors",
    "has_tile_floors",
    "total_parking",
    "total_bathrooms",
    "lot_to_living_ratio",
    "beds_per_bath",
    "has_any_pool_or_spa",
    "total_amenity_count",
    "home_type_SINGLE_FAMILY",
    "home_type_MULTI_FAMILY",
    "home_type_MANUFACTURED",
    "home_type_LOT",
    "home_type_HOME_TYPE_UNKNOWN",
    "home_type_nan",
]

# Test samples from validation set for local evaluation
TEST_SAMPLES = [
    {
        "zpid": "456753384",
        "actual_price": 308500.0,
        "features": [
            1477.0,  # living_area_sqft
            21780.0,  # lot_size_sqft
            3.0,  # bedrooms
            2.0,  # bathrooms
            30.584142684936523,  # latitude
            -87.09182739257812,  # longitude
            2025.0,  # year_built
            0.0,  # property_age
            0.0,  # days_on_zillow
            1.0,  # stories
            0.0,  # has_basement
            1.0,  # has_garage
            0.0,  # has_attached_garage
            1.0,  # has_cooling
            1.0,  # has_heating
            0.0,  # has_fireplace
            0.0,  # has_spa
            0.0,  # has_view
            0.0,  # has_pool
            0.0,  # has_open_parking
            0.0,  # has_home_warranty
            1.0,  # is_new_construction
            0.0,  # is_senior_community
            0.0,  # has_waterfront_view
            2.0,  # bathrooms_full
            0.0,  # bathrooms_half
            0.0,  # bathrooms_three_quarter
            2.0,  # garage_capacity
            2.0,  # covered_parking_capacity
            0.0,  # open_parking_capacity
            2.0,  # parking_capacity_total
            0.0,  # fireplaces_count
            2.0,  # cooling_count
            1.0,  # heating_count
            3.0,  # appliances_count
            0.0,  # flooring_count
            2.0,  # construction_materials_count
            4.0,  # interior_features_count
            0.0,  # exterior_features_count
            0.0,  # community_features_count
            2.0,  # parking_features_count
            0.0,  # pool_features_count
            2.0,  # laundry_features_count
            1.0,  # lot_features_count
            0.0,  # view_features_count
            1.0,  # sewer_count
            1.0,  # water_source_count
            2.0,  # electric_count
            3.0,  # school_count
            1.0,  # min_school_distance
            5.333333492279053,  # avg_school_rating
            7.0,  # max_school_rating
            5.5,  # elementary_rating
            7.0,  # middle_rating
            6.0,  # high_rating
            1.0,  # has_central_air
            0.0,  # has_forced_air_heating
            0.0,  # has_natural_gas
            0.0,  # has_hardwood_floors
            0.0,  # has_tile_floors
            4.0,  # total_parking
            2.0,  # total_bathrooms
            14.74610710144043,  # lot_to_living_ratio
            1.5,  # beds_per_bath
            0.0,  # has_any_pool_or_spa
            24.0,  # total_amenity_count
            1.0,  # home_type_SINGLE_FAMILY
            0.0,  # home_type_MULTI_FAMILY
            0.0,  # home_type_MANUFACTURED
            0.0,  # home_type_LOT
            0.0,  # home_type_HOME_TYPE_UNKNOWN
            0.0,  # home_type_nan
        ],
    },
    {
        "zpid": "76181016",
        "actual_price": 192900.0,
        "features": [
            1636.0,  # living_area_sqft
            12196.0,  # lot_size_sqft
            3.0,  # bedrooms
            2.0,  # bathrooms
            36.05441665649414,  # latitude
            -90.55838012695312,  # longitude
            1980.0,  # year_built
            45.0,  # property_age
            0.0,  # days_on_zillow
            1.0,  # stories
            0.0,  # has_basement
            1.0,  # has_garage
            0.0,  # has_attached_garage
            1.0,  # has_cooling
            1.0,  # has_heating
            0.0,  # has_fireplace
            0.0,  # has_spa
            0.0,  # has_view
            0.0,  # has_pool
            0.0,  # has_open_parking
            0.0,  # has_home_warranty
            0.0,  # is_new_construction
            0.0,  # is_senior_community
            0.0,  # has_waterfront_view
            2.0,  # bathrooms_full
            0.0,  # bathrooms_half
            0.0,  # bathrooms_three_quarter
            2.0,  # garage_capacity
            0.0,  # covered_parking_capacity
            0.0,  # open_parking_capacity
            1.0,  # parking_capacity_total
            0.0,  # fireplaces_count
            1.0,  # cooling_count
            1.0,  # heating_count
            6.0,  # appliances_count
            1.0,  # flooring_count
            1.0,  # construction_materials_count
            8.0,  # interior_features_count
            2.0,  # exterior_features_count
            0.0,  # community_features_count
            2.0,  # parking_features_count
            0.0,  # pool_features_count
            3.0,  # laundry_features_count
            2.0,  # lot_features_count
            0.0,  # view_features_count
            1.0,  # sewer_count
            1.0,  # water_source_count
            1.0,  # electric_count
            3.0,  # school_count
            0.699999988079071,  # min_school_distance
            7.0,  # avg_school_rating
            7.0,  # max_school_rating
            7.0,  # elementary_rating
            7.0,  # middle_rating
            5.5,  # high_rating
            0.0,  # has_central_air
            0.0,  # has_forced_air_heating
            0.0,  # has_natural_gas
            0.0,  # has_hardwood_floors
            0.0,  # has_tile_floors
            0.0,  # total_parking
            2.0,  # total_bathrooms
            7.45476770401001,  # lot_to_living_ratio
            1.5,  # beds_per_bath
            0.0,  # has_any_pool_or_spa
            33.0,  # total_amenity_count
            1.0,  # home_type_SINGLE_FAMILY
            0.0,  # home_type_MULTI_FAMILY
            0.0,  # home_type_MANUFACTURED
            0.0,  # home_type_LOT
            0.0,  # home_type_HOME_TYPE_UNKNOWN
            0.0,  # home_type_nan
        ],
    },
]
