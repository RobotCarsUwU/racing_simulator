import os

baseFile = "all_track_data_cleaned_new.csv"
previous_file = "all_track_data_cleaned.csv"

with open(baseFile, "w") as outfile:
    with open(baseFile, "r") as bf:
        outfile.write(bf.read())
    for file in os.listdir("data_not_processed"):
        filepath = os.path.join("data_not_processed", file)
        if os.path.isfile(filepath):
            with open(filepath, "r") as infile:
                outfile.write(infile.read())
