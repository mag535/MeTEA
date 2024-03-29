# MeTEA - Metagenomic Taxonomic Evaluation and Assessment Tool
## by Melissa Gray (http://github.com/mag535) and Zhengqiao Zhao (http://github.com/z2e2) for the CAMI2 Challenge benchmarking
*Parse, organize, calculate and save data from metagenomic profile files.*

  This package reads the data from profile files (with the '.profile' extension), calculates each Tax ID's confusion matrix for each tool, creates dendrograms based on taxa levels and confusion matrix metrics, then saves the confusion matrix data as an excel file and the dendrograms as png files.

  *For this package, you only need to use the main() function from the 'Misc' class in 'precall.'*

  Confusion matrices have four metrics: True Positives, False Negatives, False Positives, and True Negatives.True Positives are calculated by adding up how many samples a Tax ID is predicted to be in when it is truly there (per the ground truth profile). False Negatives are calculated by adding up the number of samples that truly have that Tax ID (per the ground truth profile) but are missing from the predicted profile. False Positives are calculated by adding up the number of samples that are predicted to have a Tax ID but are missing from the ground truth profile. True Negatives are calculated by adding up the amount of samples that a Tax ID is not predicted in and is not in the ground truth profile. This is iterated over every Tax ID in each predicted profile. Each metric of the confusion matrix is saved to a separate sheet in the Excel file.

  Precision and recall are also included in the excel file.

  Afterwards, dendrograms are created for each taxa level based on the bray curtis similarity of each tool. Those are then saved as png files in the same directory as the confusion matrix excel file.

  The ground truth file, output excel file name, input file directory for all profiles, and output file directory are all specified when calling the main() function with an object from the  ‘Misc’ class in ‘precall’ (all profile files should be in the same directory, including the ground truth). 
  For example, if your predicted profile files and the ground truth file are in a folder called “inputs” and the ground truth file is called “ground_truth.profile” and you want to save the output files to a folder called “outputs,” the command should be:
	
```python
from MeTEA.precall import Misc
Quick = Misc()
Quick.main(“ground_truth.profile”, “TaxaEvaluation_byTool”, “C:\\Users\\user\\inputs”, “C:\\Users\\user\\outputs”)
```
	

The main function takes in five arguments, three of which are optional:
- Input: the name of the ground truth profile file
- Output: the excel file name, a .xlsx of six sheets: True Positives, False Negatives, False Positives, True Negatives, Precision, and Recall of each tool
- (optional) input directory of all profiles, including the ground truth; <Default: Directory of Package Manual>
- (optional) the output directory; <Default: Directory of Package Manual>
- (optional) "yes" if you want individual .csv files of each tool's confusion matrix; <Default: "no">


A spreadsheet and heat map of the top Taxa based on difficulty and a metric can also me made.

```python
Quick.get_top_taxid(3, "tp", "easy", "yes")
Quick.create_heat_map("Top_Easy-TP_taxid.xlsx")
```

Arguments <*get_top_taxid()*>:
- Input: The number of Tax IDs to include
- (optional) The metric to evaluate by; <Default: "tp" (True Positives)>
- (optional) The difficulty level; <Default: "eas">
- (optional) "yes" if you want to only include Tax IDs present in the ground truth profile, "no" if you want to include Tax IDs found in the ground truth and predicted profiles; <Default: "yes">

The output is a .xlsx file containing the Tax IDs. It's name is formatted like this: *'Top_Difficulty-METRIC_taxid.xlsx'*

Arguments <*create_heat_map()*>:
- Input: The name of the spreadsheet containing the list of Tax IDs. They should be formatted into a column labeled 'Tax ID'.

The output is a heat map with a dnedrogram at the top. It's name is formatted as the name of the input file with the extentsion replaced with *'_Heat_Map.png'*.


You can find a quick start example and more details on the package in the jupyter note book 'Package Manual'.
