import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:            # Open the CSV file for reading
        csv_reader = csv.DictReader(csv_file)                               # Create a CSV reader object
        data = []                                                           # Initialize an empty list to store rows
        for row in csv_reader:                                              # Iterate over each row in the CSV file
            data.append(row)                                                # Append the row to the data list
    with open(json_file_path, 'w', encoding='utf-8') as json_file:          # Write the data to a JSON file
        json.dump(data, json_file, indent=4)            # Convert the data to JSON format and write it to the JSON file

input_file = "600K_USHousingProperties.csv"
output_file = "600K_USHousingProperties.json"
csv_to_json(input_file, output_file)
