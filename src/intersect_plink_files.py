# Intersection of variants when projecting data on pre-computed model


# DBDS and CPH
COVID = pd.read_csv("/users/data/pr_00006/CHB_DBDS/data/raw_data/COVID_preQC.bim", sep="\t", header=None)
COVID = COVID.set_axis(["chr", "variant", "pos", "bp", "a1","a2"], axis=1, inplace=False)

# HO

HO = pd.read_csv("/users/data/pr_00006/human_origins/data/raw_v44_origins/bed_files/origins_v44.bim", sep="\t", header=None)
HO = HO.set_axis(["chr", "variant", "pos", "bp", "a1","a2"], axis=1, inplace=False)


# Compute intersection

