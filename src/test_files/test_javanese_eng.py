import seacrowd as sc
# Load the dataset using the default config
# dset = sc.load_dataset("nusax_mt", config_name="nusax_mt_eng_jav_seacrowd_t2t", schema="seacrowd")
# Check all available subsets (config names) of the dataset
# print(sc.available_config_names("nusax_mt"))
# Load the dataset using a specific config
dset = sc.load_dataset_by_config_name(config_name="nusax_mt_eng_jav_seacrowd_t2t")
dset.save_to_disk("nusax_mt_eng_jav_seacrowd_t2t")