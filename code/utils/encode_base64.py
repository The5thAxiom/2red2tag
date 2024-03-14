import base64

input_file_path = input('input file path: ')
output_file_path = input('output file path: ')

file_data_base64 = None

with open(input_file_path, 'rb') as file:
    file_data_base64 = base64.b64encode(file.read())

with open(output_file_path, 'wb') as file:
    file.write(file_data_base64)

with open('file.mp3', 'wb') as file:
    file.write(base64.b64decode(file_data_base64))