import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def nc_plotter(dir_str, filename, variables, nGCs):
    """A plotter for Aether netCDF (nc) results

    Parameters
    ----------
    dir_str: string
        Directory string, e.g. "run.test_acheron/UA/output/"
    filename : string
        .nc filename, e.g. "3DALL_20110320_000000.nc".
    variables : list of strings
        A list of variable names for plotting
    """

    # matplotlib setting
    plt.rcParams["figure.autolayout"] = True

    # Read netCDF file
    fn = dir_str + filename
    ds = nc.Dataset(fn)

    # Print dataset to show correct reading
    print(ds)

    # Check if requested variables exists in dictionary
    var_keys = ds.variables.keys()
    variables_trim = []
    for var_str in variables:
        if var_str not in var_keys:
            print("Variable " + var_str + " does not exist in .nc file!")
        else:
            variables_trim.append(var_str)

    # Get dimensions
    nLons = len(ds.dimensions["lon"])
    nLats = len(ds.dimensions["lat"])
    #ntime = len(ds.dimensions["time"])
    nblocks = len(ds.dimensions["block"])
    nz = len(ds.dimensions["z"])

    # Get lat-lon
    lon = ds.variables["lon"][:]
    lat = ds.variables["lat"][:]

    #print(ds.variables["z"][:][0, :, :, 2])

    # Get variable data requested
    for var_str in variables_trim:
        var = ds.variables[var_str][:]

        # Get return string
        return_str = var_str + "_" + filename[:-2] + "pdf"

        # Get vmin and vmax
        vmin = int(np.nanmin(var))-1
        vmax = int(np.nanmax(var))+1

        if (var_str == "density_O"):
            vmin = 0.1
            vmax = 1000.


        ## Plot data
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        m = Basemap(projection='ortho',lon_0=315,lat_0=0)
        m.drawcoastlines()

        # Plot data in blocks
        for b in range(nblocks):
            curr_lon = lon[b, 2:-2, 2:-2, 2]
            curr_lat = lat[b, 2:-2, 2:-2, 2]
            curr_var = np.round(var[b, 2:-2, 2:-2, 2], 1)

            #if (var_str is "velocity_north_neutral"):
            #    print(curr_var)
            m.contourf(curr_lon, curr_lat, curr_var, latlon=True, vmin=vmin, vmax=vmax, levels=20)

        cmappable = ScalarMappable(norm=Normalize(vmin,vmax))
        fig.colorbar(cmappable)
        fig.savefig(return_str, transparent=True, bbox_inches='tight')

nc_plotter("run.test_acheron/UA/output/", "3DALL_20110320_000000.nc", ["density_O", "velocity_east_neutral", "velocity_north_neutral"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110321_000000.nc", ["density_O", "velocity_east_neutral", "velocity_north_neutral"], 2)
nc_plotter("run.test_acheron/UA/output/", "3DALL_20110321_000000.nc", ["density_O", "velocity_east_neutral", "velocity_north_neutral"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110321_002640.nc", ["density_O"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110322_002000.nc", ["density_O"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110323_001320.nc", ["density_O"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110324_000640.nc", ["density_O"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110325_000000.nc", ["density_O"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110320_001000.nc", ["density_O", "velocity_east_neutral", "velocity_north_neutral"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110320_001500.nc", ["density_O", "velocity_east_neutral", "velocity_north_neutral"], 2)
#nc_plotter("run.test_acheron/UA/output/", "3DALL_20110321_000000.nc", ["density_O", "velocity_east_neutral", "velocity_north_neutral"], 2)