# library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import pickle
import time
from sklearn.preprocessing import LabelEncoder

# Global Variables

# model path
model_path = 'normal_model.pkl'


# Load pickle model
forest_best = pickle.load(open(model_path, 'rb'))

# function to predict damage


def predict_damage(input_data):
    """
    input_data : This is the dictionary obtained from frontend 
    e.g.:
    input_data = {
    'count_floors_pre_eq' : 4,
    'count_floors_post_eq' : 0,
    'age_building' : 42,
    'plinth_area_sq_ft' : 420,
    'height_ft_pre_eq' : 28,
    'height_ft_post_eq' : 0,  
    'land_surface_condition' : 'Flat',
    'position' : 'Attached-1 side',
    'has_superstructure' : 'adobe_mud',
    'condition_post_eq' : 'Damage-Rubble Clear-New building built',
    'technical_solution_proposed' : 'Reconstruction',
    'foundation_type' : 'Mud mortar-Stone/Brick',
    'roof_type' : 'Bamboo/Timber-Light roof',
    'ground_floor_type' : 'Mud',
    'other_floor_type' : 'Timber/Bamboo-Mud',
    'plan_configuration' : 'Rectangular'    
    }
    """
    # dataset path
    df = pd.read_csv("csv_building_structure.csv")
    #
    df.dropna(inplace=True)
    df_web = pd.Series(input_data)
    cat_nominal = ["foundation_type", "roof_type",
                   "ground_floor_type", "other_floor_type", "plan_configuration","condition_post_eq"]
    nominal_categories = {}
    for nom in cat_nominal:
        nominal_categories[f'{nom}s'] = list(df[nom].unique())
    # Label Encoded cols
    land_surface_conditions = ['Flat', 'Moderate slope', 'Steep slope']
    positions = ['Attached-1 side', 'Attached-2 side',
                 'Attached-3 side', 'Not attached']
    damage_grades = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    # technical_solutions_proposed = [
    #     'Major repair', 'Minor repair', 'No need', 'Reconstruction']
    # conditions_post_eq = ['Covered by landslide','Damaged-Not used','Damaged-Repaired and used','Damaged-Rubble Clear-New building built','Damaged-Rubble clear','Damaged-Rubble unclear','Damaged-Used in risk','Not damaged']

    # label encoding for web variable
    df_web['land_surface_condition'] = land_surface_conditions.index(
        df_web['land_surface_condition'])
    df_web['position'] = positions.index(df_web['position'])
    # df_web['condition_post_eq'] = conditions_post_eq.index(
    #     df_web['condition_post_eq'])
    # df_web['technical_solution_proposed'] = technical_solutions_proposed.index(
    #     df_web['technical_solution_proposed'])

    # One hot encoding for web variable
    # has_superstructure, foundation_type, roof_type, ground_floor_type, other_floor_type, plan_configuration
    # * superstructure
    superstructures = ['adobe_mud', 'mud_mortar_stone', 'stone_flag', 'cement_mortar_stone', 'mud_mortar_brick',
                       'cement_mortar_brick', 'timber', 'bamboo', 'rc_non_engineered', 'rc_engineered', 'other']
    for superstructure in superstructures:
        for input_superstructure in input_data['has_superstructure']:
            key = f'has_superstructure_{superstructure}'
            if key in df_web.keys():
                if df_web[f'has_superstructure_{superstructure}'] != 1:
                    df_web[f'has_superstructure_{superstructure}'] = 1 if input_superstructure == superstructure else 0
            else:
                df_web[f'has_superstructure_{superstructure}'] = 1 if input_superstructure == superstructure else 0

    # * foundation_type
    for foundation_type in nominal_categories['foundation_types']:
        df_web[f'foundation_type_{foundation_type}'] = 1 if input_data['foundation_type'] == foundation_type else 0

    # * roof_type
    for roof_type in nominal_categories['roof_types']:
        df_web[f'roof_type_{roof_type}'] = 1 if input_data['roof_type'] == roof_type else 0

    # * ground_floor_type
    for ground_floor_type in nominal_categories['ground_floor_types']:
        df_web[f'ground_floor_type_{ground_floor_type}'] = 1 if input_data['ground_floor_type'] == ground_floor_type else 0

    # * other_floor_type
    for other_floor_type in nominal_categories['other_floor_types']:
        df_web[f'other_floor_type_{other_floor_type}'] = 1 if input_data['other_floor_type'] == other_floor_type else 0

    # * plan_configuration
    for plan_configuration in nominal_categories['plan_configurations']:
        df_web[f'plan_configuration_{plan_configuration}'] = 1 if input_data['plan_configuration'] == plan_configuration else 0

    # * plan_configuration
    for condition_post_eq in nominal_categories['condition_post_eqs']:
        df_web[f'condition_post_eq_{condition_post_eq}'] = 1 if input_data['condition_post_eq'] == condition_post_eq else 0

    df_web['net_floors'] = df_web['count_floors_post_eq'] - df_web['count_floors_pre_eq']
    df_web['net_height'] = df_web['height_ft_post_eq'] - df_web['height_ft_pre_eq']
    # removing unwanted cols
    cols_to_remove = ['has_superstructure', 'foundation_type', 'roof_type',
                      'ground_floor_type', 'other_floor_type', 'plan_configuration','condition_post_eq']
    df_test_web = df_web.drop(cols_to_remove)

    df_test_web = df_test_web.to_frame().transpose()

    reordered_cols = ['count_floors_pre_eq', 'count_floors_post_eq', 'age_building',
       'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq',
       'land_surface_condition', 'position', 'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
       'foundation_type_Bamboo/Timber', 'foundation_type_Cement-Stone/Brick',
       'foundation_type_Mud mortar-Stone/Brick', 'foundation_type_Other',
       'foundation_type_RC', 'ground_floor_type_Brick/Stone',
       'ground_floor_type_Mud', 'ground_floor_type_Other',
       'ground_floor_type_RC', 'ground_floor_type_Timber',
       'other_floor_type_Not applicable', 'other_floor_type_RCC/RB/RBC',
       'other_floor_type_TImber/Bamboo-Mud', 'other_floor_type_Timber-Planck',
       'plan_configuration_Building with Central Courtyard',
       'plan_configuration_E-shape', 'plan_configuration_H-shape',
       'plan_configuration_L-shape', 'plan_configuration_Multi-projected',
       'plan_configuration_Others', 'plan_configuration_Rectangular',
       'plan_configuration_Square', 'plan_configuration_T-shape',
       'plan_configuration_U-shape', 'condition_post_eq_Covered by landslide',
       'condition_post_eq_Damaged-Not used',
       'condition_post_eq_Damaged-Repaired and used',
       'condition_post_eq_Damaged-Rubble Clear-New building built',
       'condition_post_eq_Damaged-Rubble clear',
       'condition_post_eq_Damaged-Rubble unclear',
       'condition_post_eq_Damaged-Used in risk',
       'condition_post_eq_Not damaged', 'roof_type_Bamboo/Timber-Heavy roof',
       'roof_type_Bamboo/Timber-Light roof', 'roof_type_RCC/RB/RBC',
       'net_floors', 'net_height']
    df_test_web = df_test_web.reindex(columns=reordered_cols)
    # print(df_test_web.isna().sum())

    y_pred = forest_best.predict(df_test_web.to_numpy())

    predicted_output = damage_grades[y_pred[0]]

    return predicted_output


def find_house(building_id):
    '''
    building_id : Data obtained from the user at frontend which is their building number.

    '''
    # dataset path
    df = pd.read_csv("csv_building_structure.csv")
    #
    df.dropna(inplace=True)
    if df.loc[df['building_id'] == building_id].shape[0] > 0:
        # return df.loc[df['building_id'] == building_id].damage_grade.to_list()[0]
        # columns with numeric values | no need to encode
        numeric = ['building_id', 'district_id', 'vdcmun_id', 'ward_id', 'count_floors_pre_eq', 'count_floors_post_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq', 'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                   'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered']
        # category can either be nominal or ordinal
        cat_ordinal = ["land_surface_condition", "position", "damage_grade",
                       "technical_solution_proposed", "condition_post_eq"]
        cat_nominal = ["foundation_type", "roof_type",
                       "ground_floor_type", "other_floor_type", "plan_configuration"]
        damage_grades = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']

        df[cat_ordinal] = df[cat_ordinal].apply(LabelEncoder().fit_transform)
        df = pd.get_dummies(df, columns=cat_nominal, prefix=cat_nominal)

        X_normal = df.loc[df['building_id'] == building_id].drop(
            ['damage_grade', 'building_id', 'district_id', 'vdcmun_id', 'ward_id'], axis=1)
        y_pred_normal = forest_best.predict(X_normal.to_numpy())
        return damage_grades[y_pred_normal[0]]

    else:
        return'House not Found!!!'


if __name__ == '__main__':
    input_data = {
        'count_floors_pre_eq': 1,
        'count_floors_post_eq': 1,
        'age_building': 30,
        'plinth_area_sq_ft': 308,
        'height_ft_pre_eq': 9,
        'height_ft_post_eq': 9,
        'land_surface_condition': 'Flat',
        'position': 'Not attached',
        'has_superstructure': ['mud_mortar_stone'],
        'condition_post_eq': 'Damage-Repaired and used',
        'foundation_type': 'Other',
        'roof_type': 'Bamboo/Timber-Light roof',
        'ground_floor_type': 'Mud',
        'other_floor_type': 'Not applicable',
        'plan_configuration': 'Rectangular'
    }

    input_data1 = {
        'count_floors_pre_eq': 5,
        'count_floors_post_eq': 0,
        'age_building': 25,
        'plinth_area_sq_ft': 200,
        'height_ft_pre_eq': 30,
        'height_ft_post_eq': 0,
        'land_surface_condition': 'Flat',
        'position': 'Not attached',
        'has_superstructure': ['mud_mortar_stone', 'timber'],
        'condition_post_eq': 'Damage-Not used',
        'foundation_type': 'Mud mortar-Stone/Brick',
        'roof_type': 'Bamboo/Timber-Heavy roof',
        'ground_floor_type': 'Mud',
        'other_floor_type': 'Timber/Bamboo-Mud',
        'plan_configuration': 'Rectangular'
    }

    grade_descriptions = {
        'Grade 1': 'Hairline to thin cracks in plaster on few walls, falling of plaster bits in limited parts, fall of loose stone from upper part of the building in a few cases, only architectural repairs needed.',
        'Grade 2': 'Cracks in many walls, falling of plaster in last bits over large area, damage to non structural parts like chimney, projecting cornices. The load carrying capacity of the building is not reduced appreciably.',
        'Grade 3': 'Large and extensive cracks in most walls, collapse of small portion of non load-bearing walls, roof tile detachment, tilting or failing of chimneys, failure of individual non-structural elements such as partition/gable walls, delamination of stone/adobe walls, load carrying capacity of structure is partially reduced and significant structural repair is required.',
        'Grade 4': 'Large gaps occur in walls, walls collapse, partial structural failure of floor/roof, building takes a dangerous state.',
        'Grade 5': 'Total or near collapse of the building.'
    }

    # house_id = 120101000111
    start_time = time.time()
    predicted_grade = predict_damage(input_data1)
    predicted_description = grade_descriptions[predicted_grade]
    print(f'Predicted Damage For Input: {predicted_grade}')
    print(f'Damage Description: {predicted_description}')
    # print(find_house(house_id))
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Total Time taken: {time_taken} seconds')