import numpy as np
from astropy.io import fits

from balrogo import gaia

# Read HDU
print("Reading input...")
path = r"./samples/SCULPTOR.fits"

hdu = fits.open(path)

ra = np.asarray(hdu[1].data[:]["ra      "])  # degrees
dec = np.asarray(hdu[1].data[:]["DEC     "])  # degrees
pmra = np.asarray(hdu[1].data[:]["pmra    "])  # mas/yr
epmra = np.asarray(hdu[1].data[:]["pmra_error"])  # mas/yr
pmdec = np.asarray(hdu[1].data[:]["pmdec   "])  # mas/yr
epmdec = np.asarray(hdu[1].data[:]["pmdec_error"])  # mas/yr
corrpm = np.asarray(hdu[1].data[:]["pmra_pmdec_corr"])  # mas/yr
g_mag = np.asarray(hdu[1].data[:]["phot_g_mean_mag"])  # in mag
br_mag = np.asarray(hdu[1].data[:]["bp_rp   "])  # in mag
ast_err = np.asarray(hdu[1].data[:]["astrometric_excess_noise"])  # no units
chi2 = np.asarray(hdu[1].data[:]["astrometric_chi2_al"])  # no units
nu = np.asarray(hdu[1].data[:]["astrometric_n_good_obs_al"])  # no units
br_excess = np.asarray(hdu[1].data[:]["phot_bp_rp_excess_factor"])  # no units

wpar = np.asarray(hdu[1].data[:]["parallax"])

full_data = hdu[1].data[:]

hdu.close()

# Run sample code
print("Running code...")
idx_final, results_sd, var_sd, results_pm, var_pm = gaia.find_object(
    ra,
    dec,
    pmra,
    pmdec,
    epmra,
    epmdec,
    corrpm,
    chi2,
    nu,
    g_mag,
    br_mag,
    br_excess,
    sd_model="plummer",
    min_method="dif",
    prob_method="pm",
    prob_limit=0.9,
    use_hrd=True,
    nsig=3,
)

print("results_pm:", results_pm)
print("var_pm:", var_pm)

print("results_sd:", results_sd)
print("var_sd:", var_sd)
