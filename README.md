<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites and Installation

To get starte clone this repositry:
 ```sh
git clone https://github.com/oeberle/xai_digital_humanities.git
  ```

To make sure you have all required packages installed we provide a `requirements.txt`
 that you can install using:
 ```sh
pip3 install -r requirements.txt
  ```
After activating this environment you should be ready to use the code.


<!-- USAGE EXAMPLES -->
## A)   XAI for historical instruments

**Get started:** To reproduce the results found in our paper, you can execute the  `xai_analysis.ipynb`
notebook. Our trained model weights `model.pt` can be downloaded by exceuting `git lfs pull` in the project repository.

**Data:** We have provided sample data in the `sample_data/pages` directory. The accompanying csv-file `sample_data/data_subset.csv` contains further information about the source and label of each sample. 

**XAI**: Our explanation code for the computation and visualization of LRP heatmaps can be found in `xai.py`.  An in-depth tutorial regarding LRP can be found [here](https://git.tu-berlin.de/gmontavon/lrp-tutorial).

###


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## B)  Building your own XAI project
In case you want to adapt this code to your own research project, we next will provide some further details.

**Data Collection:** If you want to work with your own data, you can follow our data structure that enables you to directly use the provided dataloader. For this, we assume that data is organized as follows:

`<data_root>/<book_name>/raw/<book_name>_0001.jpg`

For example;
`sample_data/pages/2257_fine_sphaera_1551/raw/2257_fine_sphaera_1551_p51.jpg`
with
`data_root = sample_data/pages`
`book_name = 2257_fine_sphaera_1551`
`page_id = 2257_fine_sphaera_1551_p51.jpg`

In the accompaying csv file, you need to provide the following fields:
- **`page_id`**: Identifier composed of book_name and page number (see above), e.g. `2257_fine_sphaera_1551_p51.jpg`
- **`label`**: provide the label as an integer, from 0, 1, ..., #classes
-  **`xywh`**: if available, you can provide bounding box coordiantes, e.g.  `"[78,128,307,416]"`, otherwise just `None`

**Model**: To adapt the basic VGG model to predict the correct number of classes you have to set the number of classes `n_classes = ...` in `run_training.py`. 

**Training**: Specify the correct `data_root`  and location of your `csv-file`and start optimization by executing `python run_training.py`. Results will be saved into your specified `savedir=...` To track training performance, we use the `TensorboardX`framework. You can access the results overview  by calling `tensorboard --logdir <savedir> --port 6077 --bind_all`.

**XAI**:  After initializing your model with the resulting model weights in `<savedir>/.../model_train.pt`, you can follow the xai analysis in `xai_analysis.ipynb`. For a full xai corpus analysis, you can also adapt the `run_xai_analysis.py` script.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Oliver Eberle - [@oeberle](https://twitter.com/oeberle) 
Hassan El-Hajj - [@hassanhajj910](https://twitter.com/@hassanhajj910) 


Project Link: [https://github.com/oeberle/xai_digital_humanities.git](https://github.com/oeberle/xai_digital_humanities.git)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
