"""
Script to extract intensity values (second column) from txt files 
in the raw 900-1100 folder and organize them into a CSV file.

Author: Generated script for OP-airPLS project
Date: 2025-09-01
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

def extract_intensities_to_csv(txt_folder_path, output_csv_path):
    """
    Extract intensity values (second column) from all txt files in a folder
    and organize them into a CSV file.
    
    Parameters:
    -----------
    txt_folder_path : str
        Path to the folder containing txt files
    output_csv_path : str
        Path to save the output CSV file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Get all txt files in the folder
        txt_files = glob.glob(os.path.join(txt_folder_path, "*.txt"))
        txt_files.sort()  # Sort filenames for consistent order
        
        if not txt_files:
            print(f"No txt files found in {txt_folder_path}")
            return False
        
        print(f"Found {len(txt_files)} txt files to process:")
        for txt_file in txt_files:
            print(f"  - {os.path.basename(txt_file)}")
        
        # Dictionary to store data
        data_dict = {}
        wavenumbers = None
        
        # Process each txt file
        for i, txt_file in enumerate(txt_files):
            filename = os.path.basename(txt_file)
            filename_without_ext = os.path.splitext(filename)[0]
            
            try:
                # Load the txt file (assuming space or tab separated)
                data = np.loadtxt(txt_file)
                
                if data.shape[1] < 2:
                    print(f"Warning: {filename} does not have at least 2 columns. Skipping.")
                    continue
                
                # Extract wavenumbers (first column) and intensity (second column)
                current_wavenumbers = data[:, 0]
                intensity = data[:, 1]
                
                # Use the first file's wavenumbers as reference
                if wavenumbers is None:
                    wavenumbers = current_wavenumbers
                    data_dict['Wavenumber'] = wavenumbers
                else:
                    # Check if wavenumbers are consistent
                    if len(current_wavenumbers) != len(wavenumbers):
                        print(f"Warning: {filename} has different length ({len(current_wavenumbers)}) "
                              f"compared to reference ({len(wavenumbers)}). Attempting to align...")
                        
                        # Try to interpolate or truncate to match reference length
                        if len(current_wavenumbers) > len(wavenumbers):
                            # Truncate
                            intensity = intensity[:len(wavenumbers)]
                            print(f"  Truncated {filename} to match reference length")
                        else:
                            # Pad with zeros
                            padding_length = len(wavenumbers) - len(current_wavenumbers)
                            intensity = np.pad(intensity, (0, padding_length), mode='constant', constant_values=0)
                            print(f"  Padded {filename} with {padding_length} zeros")
                    
                    elif not np.allclose(current_wavenumbers, wavenumbers, rtol=1e-5):
                        print(f"Warning: {filename} has different wavenumbers. Using intensity values as-is.")
                
                # Store intensity values with filename as column name
                data_dict[filename_without_ext] = intensity
                
                print(f"✓ Processed {filename}: {len(intensity)} data points")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")
                continue
        
        # Create DataFrame
        if len(data_dict) <= 1:  # Only wavenumbers, no intensity data
            print("No valid intensity data found!")
            return False
        
        df = pd.DataFrame(data_dict)
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        
        print(f"\n✓ Successfully created CSV file: {output_csv_path}")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Wavenumber range: {wavenumbers[0]:.2f} - {wavenumbers[-1]:.2f}")
        
        # Display first few rows
        print(f"\nFirst 5 rows of the data:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"Error in extract_intensities_to_csv: {str(e)}")
        return False

def main():
    """Main function to run the extraction."""
    
    # Define paths
    txt_folder_path = r"E:\Desktop\OP-airPLS\3.data\raw 900-1100"
    output_csv_path = r"E:\Desktop\OP-airPLS\2. ML Part\extracted_spectra_from_txt.csv"
    
    print("=== Extracting Intensities from TXT Files to CSV ===")
    print(f"Input folder: {txt_folder_path}")
    print(f"Output file: {output_csv_path}")
    print()
    
    # Check if input folder exists
    if not os.path.exists(txt_folder_path):
        print(f"Error: Input folder does not exist: {txt_folder_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Run the extraction
    success = extract_intensities_to_csv(txt_folder_path, output_csv_path)
    
    if success:
        print("\n=== Extraction completed successfully! ===")
        
        # Provide usage information
        print("\n=== 使用说明 ===")
        print(f"1. 生成的CSV文件位置: {output_csv_path}")
        print("2. CSV文件结构:")
        print("   - 第一列: Wavenumber (波数)")
        print("   - 其他列: 各个txt文件的强度值 (列名为文件名)")
        print("3. 该CSV文件可以直接用于机器学习模型进行光谱处理")
        print("4. 如果需要修改输出路径，请编辑脚本中的 output_csv_path 变量")
        
    else:
        print("\n=== Extraction failed! ===")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
