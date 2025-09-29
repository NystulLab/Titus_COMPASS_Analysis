import os
import pandas as pd

def remove_duplicates(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            # Read the CSV file
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            
            # Remove duplicate rows based on 'source_file' and 'average_mean_intensity' columns
            df_cleaned = df.drop_duplicates(subset=['source_file', 'average_mean_intensity'])
            
            # Create the output filename
            output_filename = f"{os.path.splitext(filename)[0]}_ave.csv"
            output_file_path = os.path.join(output_folder, output_filename)
            
            # Write the cleaned dataframe to a new CSV file
            df_cleaned.to_csv(output_file_path, index=False)

input_folder = "//home/el_tito/Documents/PD_img_data/test tiff/test/stitch_out/"
output_folder = "/home/el_tito/Documents/PD_img_data/test tiff/test/stitch_out/trimmed/"
remove_duplicates(input_folder, output_folder)