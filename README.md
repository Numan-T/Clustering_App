# Clustering_App

## Information
This Project includes the source code of the Bachelor Thesis "Different Approaches to Cluster
Analysis on Sports Data" (2022) by Numan M. Tok.

## Installation
The usage of a virtual environment is recommended. Install the requirements with "pip install -r requirements.txt".

## Preparing files for Analysis
To analyze your data, first the CSV file has to be specified in the code.
The CSV file should be in the same folder as the source code.
Then, in the file "data_perparation.py" the file name of the CSV file must be specified in "csv_filename".
Don't forget to add the ending ".csv" to the filename (e.g. csv_filename="my_data.csv").
If necessary further CSV file properties can be specified in "csv_encoding", "csv_delimiter", "csv_decimal", "csv_missing_values".

## Running the Program
To run the program, first the virtual environment should be activated if one was created.
Then in the terminal, navigate to the folder with the source code and type:
"streamlit run .\streamlit_frontend.py".

The app should now be opened in your browser. If not, you can open the app by typing "http://localhost:8501/" in the browser search bar.
