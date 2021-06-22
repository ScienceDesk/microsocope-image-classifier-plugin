##  Required imports
import matplotlib; matplotlib.use('Agg')
from sdesk.proc import io
from fastai.vision.image import open_image
from fastai.basic_train import load_learner
import os
##

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Define your method main()
def main():
    # Load input file
    input_metadata = io.get_input_metadata()
    files = io.get_input_files(input_metadata)
    sdesk_input_file = files[0]
    file_metadata = input_metadata[0]

    # Load supporting file (convolution neural network)
    support_file_metadata = io.get_support_files_metadata()
    support_files = io.get_support_files(support_file_metadata)
    neural_network = support_files[0]

    # Process input file and produces results
    file_name = file_metadata['name']
    with open(sdesk_input_file.path(), 'rb') as fp:
        folder, model_filename = os.path.split(neural_network.path())
        inference_learner = load_learner(folder, model_filename)
        fastai_image_obj = open_image(sdesk_input_file.path())
        category, index, values = inference_learner.predict(fastai_image_obj)
        category_name = inference_learner.data.classes[index.item()]

    # Update custom metadata of input file using results
    file_custom_metadata = file_metadata['custom_metadata']
    file_custom_metadata['image_classification'] = category_name
    file_custom_metadata['technique'] = 'SEM'
    io.update_input_metadata(input_metadata)



# Call method main()
main()