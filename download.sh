wget -O WSD_Training_Corpora.zip http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip
wget -O WSD_Unified_Evaluation_Datasets.zip http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip
# Extract files
unzip -o WSD_Training_Corpora.zip
unzip -o WSD_Unified_Evaluation_Datasets.zip
# Create data directory
rm -f WSD_Training_Corpora.zip
rm -f WSD_Unified_Evaluation_Datasets.zip
# Move to desired location
mkdir data
mv WSD_Training_Corpora data/Training_Corpora
mv WSD_Unified_Evaluation_Datasets data/Evaluation_Datasets