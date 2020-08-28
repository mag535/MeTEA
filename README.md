# TEA
(Taxa Evaluation and Assessment) Parse, organize, calculate and save data from metagenomic profile files.

NOTE: All profile files should have the extentsion '.profile'

The easiest setup is to import Misc from precall, create a Misc object, and call it's main() function.

The main function takes in five arguments, three of which are optional:
	- Input: the name of the ground truth profile file
	- Output: the excel file name, a .xlsx of six sheets: True Positives, False Negatives, False Positives, True Negatives, Precision, and Recall of each tool
	- (optional) input directory of all profiles, including the ground truth; <Default: Directory of Package Manual>
	- (optional) the output directory; <Default: Directory of Package Manual>
	- (optional) "yes" if you want individual .csv files of each tool's confusion matrix; <Default: "no">


You can find a quick start example and more details on the package in the jupyter note book 'PAckage Manual 2.0'.
