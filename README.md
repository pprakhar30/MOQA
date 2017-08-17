# MOQA
Mixtures of Opinions for Question Answering

This is an experimental tensorflow implementation of automatically answering user posed queries on e-commerce platforms by leveraging community reviews. The model learns which reviews are relevant to queries based on community based Q/A data. It is a MoE (Mixture of Experts) based model, where when a query is posed,the reviews act as 'Experts' whose opinions are used to answer objective questions. During the training of model we learn a relevance function which helps us to rank the reviews and answer open ended questions by presenting the users with the most relevant reviews. The entire model is trained as specified in [Addressing Complex and Subjective Product-RelatedQueries with Customer Reviews][1]. This implementation specifically focusses on open ended queries.

## Requirements

- Python 2.7.6
- [Tensorflow][2]
- [NLTK][3] : for Snowball Stemmer and stop word list

## Datasets

- The Q/A community data for training the model can be downloaded from [this link][4].
- The community review data for training the model can be downloaded from [this link][5]
- The results presented below our from the model trained on **Tools and Home Improvement** and **
Musical Instruments** dataset.
- Save the both the Q/A and review data in the directory from where you want to run the script.

## Usage

To train the model and save the results:
```
python main.py qa_file.json.gz reviews_file.json.gz min_reviews k num_iter lambda MostRelevant.txt Top10Review.txt

```
* **Args**
- `qa_file.json.gz`: name of file containing the community Q/A data in the format specified in [this link][4]
- `reviews_file.json.gz`: name of file containing the community Review data in the format specified in [this link][5]
- `min_reviews`: Minimum number of reviews for an item to be considered. By default set to 1.
- `num_iter`: Number of iterations you want to train the model. By default the number of iterations is 100 as told by author in [paper][1]
- `lambda`: The regularization parameter. By default it is set to 0 i.e no regualarization.
- `MostRelevant.txt`: name of the file in which the most relevant reviews corresponding to the queries in the test dataset is stored.
- `Top10Review.txt`: name of the file in which the 10 most relevant reviews corresponding to the queries in the test dataset is stored.

##  Examples of opinions recommended by MOQA

---------------------------------------------------------------------------------
![Item 1][Item1]
**Product**: BLACK+DECKER BDCS80I 8V MAX Impact Screwdriver (amazon.com/dp/B00FFZQ0W2)

**Question**: What is the torque rating? Most impact drivers are rated in in lbs, anyone know what this is rated at? Thanks!

**Most Relevant Review**: 125 inch lbs of torque, way more than most battery powered "screwdrivers" Glad to report that the impacts also work in reverse so helpful to remove screws as well

**Actual Answer**: 125 inch pounds, which is 10.4 foot pounds. The hammer kicks in if the torque need is very high. Once a bolt or screw is loose, it quits hammering and revs up to over 2000 rpm. Sinks long drywall screws in about 2-3 seconds, no sweat. Will even sink 3 inch stair lift lag bolts. Tears down a 32 year old clothes dryer real fast.

[1]:McAuley, Julian, and Alex Yang. "Addressing complex and subjective product-related queries with customer reviews." Proceedings of the 25th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2016.(https://arxiv.org/pdf/1512.06863.pdf)
[2]:https://github.com/tensorflow/tensorflow
[3]:http://www.nltk.org/
[4]:http://jmcauley.ucsd.edu/data/amazon/qa/
[5]:http://jmcauley.ucsd.edu/data/amazon/
[Item1]:http://c.shld.net/rpx/i/s/i/spin/image/spin_prod_1012426212
