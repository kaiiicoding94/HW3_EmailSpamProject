## ADDED Requirements
### Requirement: Data Loading
The system SHALL be able to load the spam dataset from the specified CSV file.

#### Scenario: Successful data loading
- **WHEN** the data loading script is executed
- **THEN** the dataset is loaded into a pandas DataFrame.

### Requirement: Baseline Model Training
The system SHALL be able to train a logistic regression model on the pre-processed dataset.

#### Scenario: Successful model training
- **WHEN** the training script is executed
- **THEN** a trained logistic regression model is produced.
