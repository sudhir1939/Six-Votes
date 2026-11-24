# Data card

## panel_predictions.csv

1,500 multiple-choice screening questions, each answered by six AI models.

| Column | Meaning |
|---|---|
| question_id | q0001 to q1500 |
| category | python, sql, statistics, ml-concepts or system-design |
| split | dev or test (750 rows each), used only for panel selection |
| gold | the correct option: A, B, C or D |
| one column per model | the option that model chose |

The six model names are fictional and the answers are simulated. They were generated
to behave like real model outputs, including the ways real models agree and disagree.
Treat the file as you would a real evaluation log: nothing about how it was generated
is documented, and finding out what is in it is part of the job.

## digits.npz

1,797 handwritten digits as 8x8 grey-scale images (64 numbers per image, values 0 to 16),
from the UCI "Optical Recognition of Handwritten Digits" test set, as distributed with
scikit-learn. Arrays: X_train (1400, 64), y_train, X_test (397, 64), y_test.
