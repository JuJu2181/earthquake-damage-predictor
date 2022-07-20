from django.shortcuts import render
from django.http import HttpResponse
from matplotlib.style import context
from .forms import *
from earthquake.packages.predictor import predict_damage
import time

# Create your views here.
def homepage(request):
    return render(request, 'earthquake/index.html')
 
def normalUser(request):
    return render(request, 'earthquake/normal_user.html')

def professionalUser(request):
    return render(request, 'earthquake/professional_user.html')



def professionalUser_data(request):
    if request.method == "POST":
        form = professional_DataForm(request.POST)
        fl_bf_earthquake = int(form['fl_bf_earthquake'].value())
        fl_af_earthquake = int(form['fl_af_earthquake'].value())
        ht_bf_earthquake = int(form['ht_bf_earthquake'].value())
        ht_af_earthquake = int(form['ht_af_earthquake'].value())
        building_age = int(form['building_age'].value())
        plinth_area = int(form['plinth_area'].value())
        foundation_type = (form['foundation_type'].value())
        roof_type = (form['roof_type'].value())
        ground_fl_type = (form['ground_fl_type'].value())
        other_fl_type = (form['other_fl_type'].value())
        plan_configuration = (form['plan_configuration'].value())
        superstructure_type = (form['superstructure_type'].value())
        position = (form['position'].value())
        land_surface_cond = (form['land_surface_cond'].value())
        condition_post_eq = (form['condition_post_eq'].value())
        tech_soln_proposed = (form['tech_soln_proposed'].value())

        input_data = {
            'count_floors_pre_eq' : fl_bf_earthquake,
            'count_floors_post_eq' : fl_af_earthquake,
            'age_building' : building_age,
            'plinth_area_sq_ft' : plinth_area,
            'height_ft_pre_eq' : ht_bf_earthquake,
            'height_ft_post_eq' : ht_af_earthquake,  
            'land_surface_condition' : land_surface_cond,
            'position' : position,
            'has_superstructure' : list(superstructure_type),
            'condition_post_eq' : condition_post_eq,
            'technical_solution_proposed' : tech_soln_proposed,
            'foundation_type' : foundation_type,
            'roof_type' : roof_type,
            'ground_floor_type' : ground_fl_type,
            'other_floor_type' : other_fl_type,
            'plan_configuration' : plan_configuration    
        }

        print(fl_bf_earthquake)
        print(fl_af_earthquake)
        print(ht_bf_earthquake)
        print(ht_af_earthquake)
        print(building_age)
        print(plinth_area)
        print(foundation_type)
        print(roof_type)
        print(ground_fl_type)
        print(other_fl_type)
        print(plan_configuration)
        print(superstructure_type)
        print(position)
        print(land_surface_cond)
        print(condition_post_eq)
        print(tech_soln_proposed)

        start_time = time.time()
        output = predict_damage(input_data)
        print('This is the required result: ', output)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f'Total Time taken: {time_taken} seconds')

        information = {
            'Grade 1' : 'Hairline to thin cracks in plaster on few walls, falling of plaster bits in limited parts, fall of loose stone from upper part of the building in a few cases, only architectural repairs needed.',

            'Grade 2' : 'Cracks in many walls, falling of plaster in last bits over large area, damage to non structural parts like chimney, projecting cornices. The load carrying capacity of the building is not reduced appreciably.',

            'Grade 3' : 'Large and extensive cracks in most walls, collapse of small portion of non load-bearing walls, roof tile detachment, tilting or failing of chimneys, failure of individual non-structural elements such as partition/gable walls, delamination of stone/adobe walls, load carrying capacity of structure is partially reduced and significant structural repair is required.',

            'Grade 4' : 'Large gaps occur in walls, walls collapse, partial structural failure of floor/roof, building takes a dangerous state.',

            'Grade 5': 'Total or near collapse of the building'
        }
        if output in information:
            # print(information[output])

            context = {
                'output': output,
                'information': information[output]
            }


        return render(request, 'earthquake/output.html', context)