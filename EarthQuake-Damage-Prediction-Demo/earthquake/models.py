from django.db import models


# # Create your models here.


class professional_Data(models.Model):
    fl_bf_earthquake = models.IntegerField()
    fl_af_earthquake =  models.IntegerField()
    ht_bf_earthquake = models.IntegerField()
    ht_af_earthquake = models.IntegerField()
    building_age = models.IntegerField()
    plinth_area = models.IntegerField()
    foundation_type = models.CharField(max_length=50)
    roof_type = models.CharField(max_length=50)
    ground_fl_type = models.CharField(max_length=50)
    other_fl_type = models.CharField(max_length=50)
    plan_configuration = models.CharField(max_length=50)
    superstructure_type = models.CharField(max_length=50)
    position = models.CharField(max_length=50)
    land_surface_cond = models.CharField(max_length=50)
    condition_post_eq = models.CharField(max_length=50)
    tech_soln_proposed = models.CharField(max_length=50)
