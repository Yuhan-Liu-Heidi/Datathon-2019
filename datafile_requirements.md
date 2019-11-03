# Data File Requirements
* the files need to be in CSV format
* they should be saved in the same folder as the program
* to use the 1st program (machine learning), there needs to be a
  **training.csv** and a **validation.csv** data set.
  - The files need to have 4 columns containing ```userid```,
    ```inAudience```(ex. True/False for PhD), ```ltiFeatures```,
    ```stiFeatures```
  - ```ltiFeatures```, ```stiFeatures``` need to be in dictionary format
    ```{'feature A': 'percent interest'}``` (```stiFeatures``` can be an
    empty dictionary but must exist)
* to use the 2nd program (pie chart and bar graph), there needs to be an
  additional ```interest_topics.csv``` file with all feature topics in a
  column (ex. /BiomedicalEngineering/SoftwareDevelopment)
  
#### Make sure to install packages from requirements.txt to run the programs.
    
# Code Modification
If your files have names other than specified above but correct format,
change file names before running the program: 
```
# load all data
training = load("<file_name>.csv")
validation = load("<file_name>.csv")
interest_topics = pd.read_csv("<file_name>.csv")
```