# AI_Hackathon
- This is the hackathon organised by TargetHR.

## Details
- This is a 8 hour hackathon where we have to work on "Emotional Identification" problem.
- The data provided is very less data with 6 categories(different types of emotions). The training data has 900 text sentences.

## Approach:
- Preprocessed the text. The data is very unclean. I spent a lot of time processing the data.
- Converted all the tokens into integers.
- Built an LSTM model that contains Embedding layer which achieved 96.78% on train data and 39% on validation data (which was good with this data set as told by mentors.)

## Files:
- [Emotion Identification](https://github.com/Abhishekmamidi123/Emotion_identification/blob/master/Emotion_Identification.ipynb)

## Improvements that can be done:
- Text can be preprocessed more efficiently which helps us to increase accuracy.
- Instead of using Embedding layer, we can use Glove vectors for each word that were trained on large corpus.
- Improve the architecture of the model.

## Language:
- Python

## Developer:
- [M R Abhishek](https://github.com/Abhishekmamidi123)
- [LinkedIn](https://www.linkedin.com/in/abhishek-mamidi-a7a982114/)
