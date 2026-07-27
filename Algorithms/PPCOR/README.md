*This README.md file was generated on 1/19/2023 by Yiqi Su (yiqisu@vt.edu)*
<!-- remove all comments (like this) before final save  -->

# PPCOR: An R Package for a Fast Calculation to Semi-partial Correlation Coefficients

This is the instruction on how to integrate PPCOR ([[Paper](https://doi.org/10.5351/csam.2015.22.6.665)] [[GitHub](https://github.com/cran/ppcor)]) to BEELINE. Please follow the following steps:

1. **Create PPCOR folder:** Create a folder called PPCOR under Beeline/Algorithms for the new method to ensure easy set-up and portability and avoid conflicting libraries/software versions that may arise from the GRN algorithm implmentations.

2. **Create runPPCOR.R script:** In the PPCOR folder, create an R scipt runPPCOR.R to learn graph from target datasets.

3. **Create a Dockerfile:** Create a "Dockerfile" that contains necessary software specifications and commands listed in a specific order from top to bottom. 

        FROM r-base
        
        LABEL maintainer = "Aditya Pratapa <adyprat@vt.edu>"
        
        USER root
        
        WORKDIR /
        
        RUN R -e "install.packages('https://cran.r-project.org/src/contrib/ppcor_1.1.tar.gz', type = 'source')"
        
        COPY runPPCOR.R /
        
        RUN mkdir data/
        
        RUN apt-get update && apt-get install -y time

The Dockerfile will run the PPCOR algorithm within the Docker container.

4. **Add the Dockerfile to initialize.sh script:** Once the Dockerfile is ready, add the following lines to 'initialize.sh' script to create Docker image for PPCOR.

        cd $BASEDIR/Algorithms/PPCOR/
        docker build -q -t ppcor:base .
        if ([[ "$(docker images -q ppcor:base 2> /dev/null)" != "" ]]); then
            echo "Do
            cker container for PPCOR is built and tagged as ppcor:base"
        else
            echo "Oops! Unable to build Docker container for PPCOR"
        fi

5. **Create ppcorRunner.py script:** After buliding the Docker image, create a Python script called ppcorRunner.py in Beeline/BLRun folder to setup BLRun objects to read inputs and run PPCOR inside the Docker image, and also parse the output for evaluation. Specifically, the ppcorRunner.py script contains three functions:

   - ``generateInputs()`` : This function reads the expression data and applies PPCOR's mandatory gene safeguard. It keeps at most the configured ``maxGenes`` highest-variance genes and always keeps fewer genes than cells so PPCOR can provide partial-correlation p-values.
   - ``run()`` : This function constructs a "docker run" system command with parameters including the path of input data file (i.e., expression data). It also specifies where the outputs are written. The docker container run PPCOR when the parameters are passed. 
   - ``parseOutput()`` : This function reads the PPCOR-specific output and formats it into a ranked edge list (``rankedEdges.csv``) with columns Gene1, Gene2, and EdgeWeight. Gene1 and Gene2 identify the associated genes, and EdgeWeight retains the signed partial-correlation coefficient for edges passing the configured p-value cutoff. Edges are ranked by absolute magnitude without discarding their sign.

6. **Add PPCOR to runner.py:** Next, update runner.py script in Beeline/BLRun folder by adding information related to PPCOR. 

    - add "import BLRun.ppcorRunner as PPCOR"
    - add "'PPCOR':PPCOR.generateInputs" to InputMapper
    - add "'PPCOR':PPCOR.run" to AlgorithmMapper
    - add "'PPCOR':PPCOR.parseOutput" to OutputParser

7. **Add PPCOR to config.yaml:** The final step is to add the new algorithm PPCOR and any necessary parameters to the config.yaml located in Beeline/config-files folder. Note that currently BEELINE can only handle one parameter set at a time eventhough multiple parameters can be passed onto the single parameter object.

        - name: "PPCOR"
          params: 
              should_run: [True]
              # mandatory maximum gene count (also bounded below the cell count)
              maxGenes: [500]
              # p-value cutoff
              # Used in parsing output
              pVal: [0.01]
