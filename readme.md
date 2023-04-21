
# Recognizing Entity Types via Properties

This paper is accepted by [FOIS 2023](https://fois2023.griis.ca/), and the paper can be found [here](https://arxiv.org/abs/2304.07910).


### Abstract:
The mainstream approach to the development of ontologies is merging ontologies encoding different information, where one of the major difficulties is that the heterogeneity motivates the ontology merging but also limits high-quality merging performance. Thus, the entity type (etype) recognition task is proposed to deal with such heterogeneity, aiming to infer the class of entities and etypes by exploiting the information encoded in ontologies. In this paper, we introduce a property-based approach that allows recognizing etypes on the basis of the properties used to define them. From an epistemological point of view, it is in fact properties that characterize entities and etypes, and this definition is independent of the specific labels and hierarchical schemas used to define them. The main contribu- tion consists of a set of property-based metrics for measuring the contextual similarity between etypes and entities, and a machine learning-based etype recognition algorithm exploiting the proposed similarity metrics.

### files:
1. Parser: transform ontology from (N3,rdf,owl...) to triples formate and store in excel. used for OM experiments. to doc triples.
2. FCA: read triples and transform them into FCAs and encode with negative properties and layer information, including FCA-h and FCA-v. to doc FCA.
3. CreateTrainingSet: Creating Etype pairs as Training and Testing set, to doc data.
4. UtilizeFeatures: Add features on the Etype pairs, like string similarity, HS, VS, IS... to doc data/dataWithFeature.
5. model_train: train and test the model by OAEI data sets, we train on bibilo track and test on conference track.
