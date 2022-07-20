from django import forms
from .models import *


class professional_DataForm(forms.ModelForm):
    class Meta:
        model = professional_Data
        fields = ['fl_bf_earthquake', 'fl_af_earthquake', 'ht_bf_earthquake', 'ht_af_earthquake', 'building_age',
                    'plinth_area', 'foundation_type', 'roof_type', 'ground_fl_type', 'other_fl_type', 'plan_configuration',
                    'superstructure_type', 'position', 'land_surface_cond', 'condition_post_eq', 'tech_soln_proposed']