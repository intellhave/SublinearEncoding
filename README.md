# Introduction 

This is the demo code for the method described in the following paper:

* Huu Le, Michael Miford, "Large-scalve Visual Place Recognition with Sub-linear Storage Growth".  Under review for RA-L and ICRA2019.  A pdf version of the paper  can be found [here](https://www.dropbox.com/s/09rwc1eiu7juitb/paper_pdf.pdf?dl=0).

- Abstract: Robotic and animal mapping systems share many of the same objectives and challenges, but differ in one key aspect: where much of the research in robotic mapping has focused on solving the data association problem, the grid cell neurons underlying maps in the mammalian brain appear to intentionally break data association by encoding many locations with a single grid cell neuron. One potential benefit of this intentional aliasing is both sub-linear map storage \textit{and} computational requirements growth with environment size, which we demonstrated in a previous proof-of-concept study that detected and encoded mutually complementary co-prime pattern frequencies in the visual map data. In this research, we solve several of the key theoretical and practical limitations of that prototype model and achieve significantly better sub-linear storage growth, a factor reduction in storage requirements per map location, scalability to large datasets on standard compute equipment and improved robustness to environments with visually challenging appearance change. These improvements are achieved through several innovations including a flexible user-driven choice mechanism for the periodic patterns underlying the new encoding method, a parallelized chunking technique that splits the map into sub-sections processed in parallel and a novel feature selection approach that selects only the image information most relevant to the encoded temporal patterns. We evaluate our techniques on two large benchmark datasets with comparison to the previous state-of-the-art system, as well as providing detailed analysis of system performance with respect to parameters such as required precision performance and the number of cyclic patterns encoded.

# Usage
This code was tested on an Ubuntu Machine with Python 2.7. Please follow the following steps to run the demo:

* Download compressed data file at: 
https://www.dropbox.com/s/ylmwak1ncfhvnxo/data.zip?dl=0

* Extracted data.zip into the folder named "data" located at the working directory.

* Run `python demo.py`


