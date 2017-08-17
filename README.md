# MOQA
Mixtures of Opinions for Question Answering

This is an experimental tensorflow implementation of automatically answering user posed queries on e-commerce platforms by leveraging community reviews. The model learns which reviews are relevant to queries based on community based Q/A data. It is a MoE (Mixture of Experts) based model, where when a query is posed,the reviews act as 'Experts' whose opinions are used to answer objective questions. During the training of model we learn a relevance function which helps us to rank the reviews and answer open ended questions by presenting the users with the most relevant reviews. The entire model is trained as specified in [Addressing Complex and Subjective Product-RelatedQueries with Customer Reviews][1]  

# References
[1]:McAuley, Julian, and Alex Yang. "Addressing complex and subjective product-related queries with customer reviews." Proceedings of the 25th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2016.(https://arxiv.org/pdf/1512.06863.pdf)


