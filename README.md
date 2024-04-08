# Question Answering with Retrieval Augmented Generation

Add abstract here

## Requirements

We have provided a `requirements.txt` file which lists the required libraries. You can install them by running:

```bash
pip install -r requirements.txt
```

## Usage

There are two notebooks in this repository:

1. make_dataset.ipynb
2. fine_tune_rag.ipynb

We recommended running the notebooks in the order listed above. make_dataset.ipynb is used to acquire the index data from wikipedia while fine_tune_rag.ipynb is used to fine-tune the RAG model and perform some simple analysis.

## Points of improvement

We have not extensively tested the effect of parameter size on the model's performance. We make use of a small encoder ALBERT trained on a paraphrasing task. However, the original authors used a BERT model trained on a Dense Passage Retrieval task. We are not certain how this difference in the encoder affects the model's performance.

Additionally, our loss function is perhaps not the best choice for this task. We currently compute the loss using the first reference answer in the list of answers. Nonetheless, multiple answers can be correct for a given question. It is not clear how the authors of the original paper handled this issue with the WebQuestions dataset. We recommend a loss function which searches the entire list of answers for the correct answer and then computes the loss. In this case, we expect the validation loss to also decrease steadily like the training loss.

## Resources

We recommend reading the original paper by Lewis et al. (2020) which can be found [here](https://arxiv.org/abs/2005.11401).

The original authors have also provided a repository which can be found [here](https://github.com/huggingface/transformers/blob/main/examples/research_projects/rag/README.md)