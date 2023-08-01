# Counterfactual Explanations for Bias Recommendation
Code and data for the paper: "Counterfactual explanations for Bias Recommendation".

## Datasets
Folder [datasets](https://github.com/lezaf/BiasExplain/tree/main/datasets) contains the real and synthetic data we used. Within this folder:

### Real
Zipped folder ml-100k contains the public MovieLens dataset with 100K ratings.

### Synthetic
Zipped folder synthetic contains the data we generated with different parameters (in the file name). Files *.info* contain metadata for the synthetic dataset. Files *.edges* contain the generated user-item graph in the format: `<user_id_x> <item_id_y>\n` (per line), meaning `<user_id_x>` rated `<item_id_y>`.

*Synthetic dataset filename explanation:* in e.g. `synth_0.7_b_1.3_p.edges`, 0.7 is the *bias* and 1.3 is the *popularity*.

For more details in synthetic datasets generation, see [synthetic_gen.py](https://github.com/lezaf/BiasExplain/blob/main/src/utils/synthetic_gen.py).

## Code

Folder [src](https://github.com/lezaf/BiasExplain/tree/main/src) contains the code for our algorithms.

## References:

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
- Athanasios N. Nikolakopoulos and George Karypis. 2019. RecWalk: Nearly Uncoupled Random Walks for Top-N Recommendation. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining (WSDM '19). Association for Computing Machinery, New York, NY, USA, 150–158. https://doi.org/10.1145/3289600.3291016
