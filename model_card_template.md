# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a classification model based on publically available Census data.  It utilizes scikit-learn's Random Forest Classifier with default hyperparameters.

## Intended Use
This model classifies segments of the population on the basis of Salary. Salary is predicted to be either >50K or <=50K.

## Training Data
Training data was provided from the data/census.csv file with 32,562 records and the following columns:
    age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,salary
80% of the data from census.csv was used for training.

## Evaluation Data
The evaluation data represents the complimentary 20% of the census.csv data that was excluded from the training data.

## Metrics
Precision, Recall, and F1 Score are used to evaluate the model's performance.  Utilize a random_state of 69 for reproducability.

Precision: 0.7275 | Recall: 0.6129 | F1: 0.6653

## Ethical Considerations
While demographic information is used for the model (including age, education, marital-status, race, and sex), no uniquely personally identifying information (such as name, address, phone number, ssn) is present.

## Caveats and Recommendations
Any biases inherent in how this data was collected or compiled may result in biased model results.  Biases in external socio-economic systems (including but not limited to racial and educational disparities), trickling down into the data, may also be reflected in model performance.
