# Read the provided dependencies from the text file
with open('paste.txt', 'r') as file:
    dependencies = file.readlines()

# Extract package names and versions
parsed_dependencies = [line.strip() for line in dependencies if line.strip()]

# Create a requirements.txt file from the parsed dependencies
with open('requirements.txt', 'w') as file:
    for dependency in parsed_dependencies:
        # Write each dependency to the file
        file.write(dependency + '\n')

