from encoder import AttributeExtractor
from utils import uncompress_string_image
import pandas as pd
import csv


# Load pandas dataframe
df = pd.read_csv("./dataset/test.csv")

# Uncompress cropped image
df["uncompressed_feature_vector"] = df.apply(lambda x: uncompress_string_image(
                                             compresed_cropped_image=x["feature_vector"]),
                                             axis=1)

# Declare an encoder object
# Declare model variables and constants
model_weight_path = "./weights/segmentation_weights.pth"
RESIZE_DIMENSIONS = (224, 224)  # By default
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Declare an object of the class AttributeExtractor
# (Model initialization)
attribute_extractor = AttributeExtractor(
    weight_path=model_weight_path,
    resize_dimensions=RESIZE_DIMENSIONS,
    mean=MEAN,
    std=STD
)

# Add the new column
df["attributes_vector"] = attribute_extractor.get_features(list(df["uncompressed_feature_vector"]))

# Clean the dataframe
df.drop("uncompressed_feature_vector", axis=1, inplace=True)

# Write the dataframe to a .csv
df.to_csv("./output_files/output_dataset.csv",
          index=False,
          quoting=csv.QUOTE_NONNUMERIC
          )
