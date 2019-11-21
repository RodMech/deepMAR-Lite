# Deep MAR Lite

## Contents

This project is an updated version of [Deep-MAR](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch).

The version of PyTorx has been updated to a more recent one. 

The net is able to work both in GPU and CPU. 


## Requirements

The required packages have been version-pinned in the *requirements.txt*.

`torch==1.2`
`numpy==1.16.4`
`Pillow`
`pandas`


## Dockerfile

This version includes two Dockerfiles: 

- `dev-nogpu.dockerfile`
- `dev-gpu.dockerfile`: CUDA 10 // CUDNN 7

For testing the GPU container, after building it, the user could use the following command: 

`docker run --runtime=nvidia  -it --volume /home/ec2-user/deepMAR:/opt/project deepMAR-dev:gpu /bin/bash`

Or a docker-compose file could be used as well. 


## Weights

Default weights are provided by the author at [this issue](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch/issues/1). 



## How to run it

The script main.py has the parameters that the user can modify. Unfortunately, in the present version, parameters have to be hard coded in the script.

A future implementation will include a better way of changing the parameters. 

The net ingests cropped images, detections, bounding boxes with a requested format. The format is shown in the 1<sup>st</sup> point.



1.- Load a pandas dataframe:

    # Load pandas dataframe
    df = pd.read_csv("./dataset/input_dataset.csv")

One of the columns has to be a string of a dict with the following format

    img_dict = {
        "width": image_width,
        "height": image_height,
        "colors": colors,
        "image": zlibed.hex()
    }

Where the image "zlibed", in bytes format, has to be converted to hexadecimal format. 



2.- Uncompress the cropped image:

    # Uncompress cropped image
    df["uncompressed_feature_vector"] = df.apply(lambda x: uncompress_string_image(
        compresed_cropped_image=x["feature_vector"]),
        axis=1)



3.- Declare the encoder object, with its correspondent variables and constants. 

    # Declare an encoder object
    # Declare model variables and constants
    model_weight_path = "./weights/segmentation_weights.pth"
    RESIZE_DIMENSIONS = (224, 224)  # By default
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Declare an object of the class AttributeExtractor (Model initialization) 
    attribute_extractor = AttributeExtractor(
        weight_path=model_weight_path,
        resize_dimensions=RESIZE_DIMENSIONS,
        mean=MEAN,
        std=STD
    )


4.- Register the new column as a new dataframe feature:      

    # Add the new column
    df["feature_vector"] = attribute_extractor.get_features(list(df["uncompressed_feature_vector"]))
    # Clean the dataframe
    df.drop("uncompressed_feature_vector", axis=1, inplace=True)


5.- Save the data as a .csv: 

    # Write the dataframe to a .csv
    df.to_csv("./output_files/output_dataset.csv",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
    )



## Performance & AMI

The attached dockerfiles have been tested on an Amazon Linux AMI 2.0.20190614 x86_64 ECS HVM GP2.

| CPU (2,2 GHz Intel Core i7):  | GPU (Tesla K80) |
| ------------- | ------------- |
| 10 Hz  | 54 Hz  |


## Citation


    @inproceedings{li2015deepmar,
        author = {Dangwei Li and Xiaotang Chen and Kaiqi Huang},
        title = {Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios},
        booktitle = {ACPR},
        pages={111--115},
        year = {2015}
    }