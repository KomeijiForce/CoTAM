## Chain-of-Thoughts Attribute Manipulation (CoTAM)
**CoTAM** is an LLM-based framework that generates efficient training data for smaller language models.

# Run the code!
(Currently only SST-2 example is available, we will upload a unified version later)
Generate (You have to put your OpenAI API key in constant.py):
```
python cotam.py
```
Fine-tuning
```
python tune.py
```
Nearest Centroid (NC)
```
python nc.py
```
K-Nearest Neighbors
```
python knn.py
```
