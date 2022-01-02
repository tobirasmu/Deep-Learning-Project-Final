# Deep-Learning-Project
The repository for the Deep Learning Project fall 2021. Group members:

 - Sophia Elizabeth Bardenfleth
 - Nikolaj Normann Holm
 - Oskar Bohn Lassen
 - Tobias Engelhardt Rasmussen
 - Mads Thorstein Roar Henriksen 

## Project description
Project article can be found [here](https://www.overleaf.com/project/618948b35ad679e86568307c), the web-page for manually annotating frames can be found [here](https://www.makesense.ai/).

## Project Structure

       ├── CopActNet
       │   ├── analysis                         <- Code for running the models and analysing outputs
       │   ├── models                           <- Models used and trained in the project
       │   ├── preprocessing                    <- Code for preprocessing the data
       │   │   └── load_and_pickle_videos.py
       │   ├── tools                            <- Tools used in the project (reading, writing, preprocessing)
       │   │   ├── reader.py                         <- Code for reading / loading data
       │   │   ├── writer.py                         <- Code for writing / saving data
       │   │   ├── visualizer.py                     <- Code for visualizing results, show frames, etc.
       │   │   └── trainer.py                        <- Functions for to help training
       │   ├── README.md                        <- Description of the code of the CopActNet package
       │   └── config.py                        <- Configurations used in the project (paths, sizes, etc.)
       │
       ├── Data                                 <- Data used in the project (will be .gitignored)
       ├── Random                               <- Random code / sandbox
       └── README.md                            <- This file

