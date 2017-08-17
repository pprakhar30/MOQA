# MOQA
Mixtures of Opinions for Question Answering

This is an experimental tensorflow implementation of automatically answering user posed queries on e-commerce platforms by leveraging community reviews. The model learns which reviews are relevant to queries based on community based Q/A data. It is a MoE (Mixture of Experts) based model, where when a query is posed,the reviews act as 'Experts' whose opinions are used to answer objective questions. During the training of model we learn a relevance function which helps us to rank the reviews and answer open ended questions by presenting the users with the most relevant reviews. The entire model is trained as specified in (with one important change that the model is trained in batches of Items) [Addressing Complex and Subjective Product-RelatedQueries with Customer Reviews][1]. This implementation specifically focusses on open ended queries.

## Requirements

- Python 2.7.6
- [Tensorflow][2]
- [NLTK][3] : for Snowball Stemmer and stop word list

## Datasets

- The Q/A community data for training the model can be downloaded from [this link][4].
- The community review data for training the model can be downloaded from [this link][5]
- The results presented below are from the model trained on **Tools and Home Improvement** and **Musical Instruments** dataset.
- Save both the Q/A and review data in the directory from where you want to run the script.

## Usage

To train the model and save the results:
```
python main.py qa_file.json.gz reviews_file.json.gz min_reviews k num_iter lambda MostRelevant.txt Top10Review.txt

```
- **Args**
     * `qa_file.json.gz`: name of file containing the community Q/A data in the format specified in [this link][4]
     * `reviews_file.json.gz`: name of file containing the community Review data in the format specified in [this link][5]
     * `min_reviews`: Minimum number of reviews for an item to be considered. By default set to 1.
     * `num_iter`: Number of iterations you want to train the model. By default the number of iterations is 100 as told by author in [paper][1]
     * `lambda`: The regularization parameter. By default it is set to 0 i.e no regualarization.
     * `MostRelevant.txt`: name of the file in which the most relevant reviews corresponding to the queries in the test dataset is stored.
     * `Top10Review.txt`: name of the file in which the 10 most relevant reviews corresponding to the queries in the test dataset is stored.

## Implementation Details

- The model was trained on R4 High-Memory Large amazon ec2 instance.
- The model was trained for 100 epochs.
- The model was trained using Adam Optimizer rather than lbfgs-b optimizer as used in the [paper][1].
- The training is done in batches of items.
- The results presented here do not use any regularization as the number of intsances to train >> the number of features
- The reviews and Q/A fed to the model for training is pre-processed
    * Tokeninzing using nltk RegexpTokenizer.
    * Filtering out the stop words. 
    * Stemming the words used using Snowball Stemmer
 - Used Binary Mask while finding low rank bilinear embeddings rather than the feature vector as used in [paper][1].
 
## Results

| Dataset/Model             | Model A   | Model B | Model C|
|---                        |---        |---      |---     |
|Musical Instruments        | 0.657     | 0.678   | 0.729  |
|Tools and Home Improvement | 0.721     | 0.745   | 0.792  |

- `Model A`: It is the model as described in [paper][1] (except the training is done in batches of items which helps in vectorizing the loss function)
- `Model B`: In this model we use binary mask to represent question, answer and review rather than feature vector to learn bilinear embeddings.
- `Model C`: This model extends `Model B` by doing some preprocessing on the Q/A and reviews. 
    * Tokeninzing using nltk RegexpTokenizer.
    * Filtering out the stop words. 
    * Stemming the words used using Snowball Stemmer

The performance metrics used is AUC (the model's ability to assign the highest possible rank to the correct answer).


##  Examples of opinions recommended by MOQA

---------------------------------------------------------------------------------
<img src="http://c.shld.net/rpx/i/s/i/spin/image/spin_prod_1012426212" width="200" height="200">

**Product**: BLACK+DECKER BDCS80I 8V MAX Impact Screwdriver (www.amazon.com/dp/B00FFZQ0W2)

**Question**: What is the torque rating? Most impact drivers are rated in in lbs, anyone know what this is rated at? Thanks!

**Most Relevant Review**: 125 inch lbs of torque, way more than most battery powered "screwdrivers" Glad to report that the impacts also work in reverse so helpful to remove screws as well

**Actual Answer**: 125 inch pounds, which is 10.4 foot pounds. The hammer kicks in if the torque need is very high. Once a bolt or screw is loose, it quits hammering and revs up to over 2000 rpm. Sinks long drywall screws in about 2-3 seconds, no sweat. Will even sink 3 inch stair lift lag bolts. Tears down a 32 year old clothes dryer real fast.

----------------------------------------------------------------------------------------
<img src="https://images-na.ssl-images-amazon.com/images/I/51Souz9ZD9L._SL500_AC_SS350_.jpg" width="200" height="200">

**Product**: Sengoku Portable Kerosene Heater (www.amazon.com/dp/B000KKO33A)

**Question**: can heater be used for cooking

**Most Relevant Review**: After languishing in the garage untended the batteries in the quick-start had corroded but it is easy to light with a match and we ran this to keep warm, boil water, warm some food and even cook a beef stew from scratch ingredients from our failing fridge (using the ceramic insert from a slow cooker) after removing the top safety cage

**Actual Answer**: No. absolutely not. this is a space heater to warm a small room in a house.

----------------------------------------------------------------------------------------------
<img src="http://ecx.images-amazon.com/images/I/51xQCcjjMFL.jpg" width="200" height="200">

**Product**: 3M Interior Transparent Weather Sealing Tape (www.amazon.com/dp/B0000CBIFF)

**Question**: What is the best method for removing this from factory painted metal window trim and vinyl windows? I do not want to be able to tell it was there. You see, I snuck it past my husband who would refuse to risk tape on windows..:)

**Most Relevant Review**: When removing tape, remove very slowly and evenly if applied to acrylic latex painted surfaces

**Actual Answer**: It's less sticky than packing tape so shouldn't be a problem, unless the paint is cracking and peeling already. There should be no adhesive residue on the window. If there is and you don't have something like GooGone, I've heard peanut butter works. But be careful of putting anything oily on matte finishes.

--------------------------------------------------------------------------------------------------
<img src="https://images-eu.ssl-images-amazon.com/images/I/51BJ4xNpoFL._SX342_QL70_.jpg" width="200" height="200">

**Product**: Custom Leathercraft 125M Handyman Flex Grip Work Gloves (www.amazon.com/dp/B0002YPZKY)

**Questions**: Working with some pallet wood and really want to have a good grip and avoid splinters.These gloves be good for that or should I go with real leather?

**Top Most Relevant Reviews**:

- While wearing the Custom Leathercraft Flex Grip Work Gloves for DIY carpentry work, I can still use my fingers to pick up small objects, manipulate car keys etc,  There are thicker gloves on the market which provide better protection however the are usually so stiff that manual dexterity is compromised.
- If you're worried about protection, don't; they will protect just as well as any pair of leathers that I've used (the padding is synthetic leather).

**Actual Answer**: I would use the thicker gloves by CLC, 160 Contractor XC. I have both kinds of these gloves and they are a better protection against splinters.

----------------------------------------------------------------------------------------------------------
<img src="https://images-na.ssl-images-amazon.com/images/I/414IQpo6OvL._SR600,315_SCLZZZZZZZ_.jpg" width="200" height="200">

**Product**: Portable Digital Recorder (www.amazon.com/dp/B004TE5HBU)

**Questions**: Im wondering if I can wear this and connect it to an 1/8\" headset or lav to record myself doing yoga classes. will handling noise be a problem?

**Top Most Relevant Reviews**:

- I still don't know how I will solve the  wind noise issue when recording ambient sound (microphone capsules facing out) since the windscreen will only cover the microphones in the XY pattern position
- Buyers should bear in mind that the unit records ambient sounds (e.g., audience and any other background noise), so if you want a more pure sound, you should record directly to a computer through a mixer or other interface to instruments, microphones, etc.

**Actual Answer**: No, I wouldn't wear it, definitely will produce handling noise even with the headset or lav as the mic cable brushes your clothing while changing positions. Look into an inexpensive boundary mic from Radio Shack or other source, set it on the ground near you. It has a hemispherical pickup pattern, they're commonly used for conference table meetings because of this.

-----------------------------------------------------------------------------------------------------------------
## TODO

- Train the model using all the items rather than using batches of item as described in [paper][1]
- Model ambiguity by including multiple answers to a question as decribed in [paper][6]

## References

- ["Addressing complex and subjective product-related queries with customer reviews", Julian McAuley, Alex Yang World Wide Web (WWW), 2016][1]
- ["Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems", Mengting Wan, Julian McAuley International Conference on Data Mining (ICDM), 2016][6]

## License
MIT


[1]:https://arxiv.org/pdf/1512.06863.pdf
[2]:https://github.com/tensorflow/tensorflow
[3]:http://www.nltk.org/
[4]:http://jmcauley.ucsd.edu/data/amazon/qa/
[5]:http://jmcauley.ucsd.edu/data/amazon/
[6]:http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm16c.pdf
