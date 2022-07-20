from django.contrib import admin
from .models import *
# Register your models here.


# admin.site.register(normal_Data)
# admin.site.register(professional_Data)
# admin.site.register(hi)

@admin.register(professional_Data)
class professional_DataAdmin(admin.ModelAdmin):
    list_display = ['fl_bf_earthquake', 'fl_af_earthquake', 'ht_bf_earthquake', 'ht_af_earthquake', 'building_age',
                    'plinth_area', 'foundation_type', 'roof_type', 'ground_fl_type', 'other_fl_type', 'plan_configuration',
                    'superstructure_type', 'position', 'land_surface_cond', 'condition_post_eq', 'tech_soln_proposed']