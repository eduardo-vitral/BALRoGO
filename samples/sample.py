from balrogo import gaia

# Read HDU
print("Reading input...")
path = r"./samples/SCULPTOR.fits"

# Run sample code
print("Running code...")
final_data, results_sd, var_sd, results_pm, var_pm = gaia.extract_object(path)

print("results_pm:", results_pm)
print("var_pm:", var_pm)

print("results_sd:", results_sd)
print("var_sd:", var_sd)
