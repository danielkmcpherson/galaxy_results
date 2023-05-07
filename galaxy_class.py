"""
NAME:
	galaxy_class.py

AUTHOR:
    Daniel McPherson
    Swinburne
    2023

EMAIL:
	<dmcpherson@swin.edu.au>

PURPOSE:
	To create a class to define all the attributes of the galaxies
	Written on Windows 11, with Python 3.9

"""
# from astropy.io import fits

# define the galaxy class we'll use throughout our results code
class Galaxy:

    def __init__(self, gal_name, red_data_filepath, red_var_filepath, blue_data_filepath, blue_var_filepath, z, emission_lines_fit, emission_lines_folder, outflow_region_filepath, galaxy_region, galaxy_center, results_folder):
        """
        
        Create the galaxy class that we'll use throughout our results code.
        
        Parameters
        ----------
        gal_name: str
            Galaxy name or descriptor

        red_data_filepath: str
            Path to red galaxy data file

        red_var_filepath: str
            Path to red galaxy variance file

        blue_data_filepath: str
            Path to blue galaxy data file

        blue_var_filepath: str
            Path to blue galaxy variance file

        z: float
            Galaxy redshift

        emission_lines_fit: list
            List of emission lines fit for galaxy. Each item in the list should be a string
            with the emission line name (one of "Hbeta", "Hgamma", "5007", "4959", and "3727") 

        emission_lines_folder: str

        
        results_folder:
        galaxy_center:
        """