## ADDED Requirements

### Requirement: Spam Prediction UI
The system SHALL provide a web-based user interface for classifying new text messages as spam or ham.

#### Scenario: Classify a ham message
- **GIVEN** the trained model and vectorizer are loaded
- **WHEN** a user enters the text "I'll see you at the meeting tomorrow" into the text input area
- **AND** the user clicks the "Predict" button
- **THEN** the system SHALL display the result "Ham".

#### Scenario: Classify a spam message
- **GIVEN** the trained model and vectorizer are loaded
- **WHEN** a user enters the text "Congratulations you won a free cruise to the Bahamas click here" into the text input area
- **AND** the user clicks the "Predict" button
- **THEN** the system SHALL display the result "Spam".

### Requirement: Model Performance Visualization
The system SHALL display the model's evaluation metrics and confusion matrix within the web interface.

#### Scenario: View performance metrics
- **GIVEN** the application is running
- **WHEN** a user loads the page
- **THEN** the system SHALL display the model's accuracy, precision, recall, and F1-score.
- **AND** the system SHALL display the confusion matrix as a plotted image.
