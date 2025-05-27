from astropy.time import Time, TimeDelta
from pathlib import Path
import subprocess
import numpy as np
import astropy.units as u
import asdf

output_dir = Path("/grp/roman/RFP/DEV/sim_inflight_calplan/romanisim_darks")

optical_element = 'F213'  # cant simulate "DARK optical element in romanisim for some reason
ma_table_number = 18
seed = 44
level = 1
truncate = 46
cal_level = 'cal' if level == 2 else 'uncal'

# Initialize start time for short darks
obs_time = Time('2026-10-01T00:00:00')

short_dark_program = '00444'
# Simulate for detectors 1 through 1 (adjust range as needed)
for det in range(1, 2):
    sca = det  # Detector number (1–18)
    
    # Reset time for this detector
    current_time = obs_time.copy()

    for exp in range(1, 7):  # Exposures increment
        exp_str = f"{exp:04d}"
        sca_str = f"wfi{sca:02d}"
        
        filename = output_dir / f"r{short_dark_program}01001001001004_{exp_str}_{sca_str}_{optical_element.lower()}_{cal_level}.asdf"
        
        command = [
            "romanisim-make-image",
            "--date", current_time.isot,
            "--nobj", "0",
            "--sca", str(sca),
            "--level", str(level),
            "--ma_table_number", str(ma_table_number),
            "--truncate", str(truncate),
            str(filename)
        ]

        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error with exposure {exp}: {result.stderr}")
        else:
            print(f"Success: {filename.name}")

        # Add 150 seconds for the next exposure
        current_time += 150 * u.s

        with asdf.open(filename, mode='rw') as af:
            meta = af.tree["roman"]["meta"]

            # Update some meta to make consistent with what we expect to see in the dark calibration program
            meta["instrument"]["optical_element"] = "DARK" 
            # need to check that filename for darks is going to be in file string and meta https://stsci-docs.stsci.edu/display/DRAFTSOC/.Data+Levels+and+Products+v2025
            meta["exposure"]["ma_table_name"] = "DIAGNOSTIC"

            #TODO work on figuring out why this gives failed tag error
            #start_time = meta["exposure"]["start_time"]
            #new_file_date = start_time + TimeDelta(1, format="jd")  # Add 1 day
            #meta["file_date"] = new_file_date.isot

            af.update()


check_plots = False
if check_plots:
    import asdf
    import matplotlib.pyplot as plt
    import numpy as np

    filename = "r0003201001001001004_0003_wfi01_f213_uncal.asdf"

    with asdf.open(filename) as af:
        data = np.asarray(af.tree["roman"]["data"])

    # Look at read 0
    image = data[0, :, :]

    plt.figure(figsize=(10, 8))
    plt.imshow(image, origin="lower", cmap="gray",
            vmin=np.percentile(image, 2), vmax=np.percentile(image, 98))
    plt.colorbar(label="DN")
    plt.title("Integration 0, Group 0")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.tight_layout()
    plt.show()

    # Histogram in log-log space
    pixel_values = image.flatten()
    # Keep only positive values for log-log plot
    positive_values = pixel_values[pixel_values > 0]
    counts, bins = np.histogram(positive_values, bins=2000)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Plot in log-log
    plt.figure(figsize=(8, 6))
    plt.loglog(bin_centers, counts, drawstyle='steps-mid', color='blue')
    plt.title("Log-Log Histogram of Pixel Values")
    plt.xlabel("DN")
    plt.ylabel("Number of Pixels")
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()


"""
This notebook has how to run from python 
https://github.com/spacetelescope/roman_notebooks/blob/main/content/notebooks/romanisim/romanisim.ipynb
"""

